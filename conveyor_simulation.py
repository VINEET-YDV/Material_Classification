import os
import time
import csv
import datetime
import glob
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# --- 1. Configuration ---
# --- (YOU MUST CHANGE THESE PATHS) ---
ONNX_PATH = 'waste_classifier.onnx'
# Folder containing the images to be processed by the conveyor
IMAGE_FOLDER_PATH = r'C:\Users\admin\OneDrive\Desktop\Alfastack\test_data' 
RESULTS_CSV_PATH = 'conveyor_log.csv'

# --- (YOU CAN TUNE THESE SETTINGS) ---
SIMULATION_DELAY_SECONDS = 2  # Time in seconds between processing each image
CONFIDENCE_THRESHOLD = 0.75 # A value between 0.0 and 1.0
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# --- 2. Image Preprocessing ---
# These transformations MUST be the same as the ones used during training.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. Helper Functions ---
def softmax(x):
    """Compute softmax values for a set of scores."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def classify_frame(ort_session, image_path):
    """
    Takes an ONNX session and an image path, returns predicted class and confidence.
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None
        
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    ort_inputs = {ort_session.get_inputs()[0].name: input_batch.numpy()}
    
    ort_outs = ort_session.run(None, ort_inputs)
    scores = ort_outs[0][0]
    
    probabilities = softmax(scores)
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx]
    predicted_class = CLASS_NAMES[predicted_idx]

    return predicted_class, confidence

# --- 4. Main Simulation ---
if __name__ == "__main__":
    # Load the ONNX model
    try:
        ort_session = ort.InferenceSession(ONNX_PATH)
        print("ONNX model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load ONNX model from {ONNX_PATH}. Error: {e}")
        exit()

    # Find all images in the source folder
    image_paths = glob.glob(os.path.join(IMAGE_FOLDER_PATH, '*.jpg'))
    image_paths += glob.glob(os.path.join(IMAGE_FOLDER_PATH, '*.png'))

    if not image_paths:
        print(f"FATAL: No images found in {IMAGE_FOLDER_PATH}. Exiting.")
        exit()

    # Prepare the CSV file and write the header
    with open(RESULTS_CSV_PATH, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ['timestamp', 'image_name', 'predicted_class', 'confidence', 'flag']
        csv_writer.writerow(header)

        print("\n--- Starting Conveyor Belt Simulation ---")
        
        # Process each image
        for i, image_path in enumerate(image_paths):
            print(f"\n[Frame {i+1}/{len(image_paths)}] Capturing image: {os.path.basename(image_path)}")
            
            # Get prediction
            predicted_class, confidence = classify_frame(ort_session, image_path)
            
            if predicted_class is None:
                continue

            # Check confidence and set flag
            flag = ''
            if confidence < CONFIDENCE_THRESHOLD:
                flag = 'LOW CONFIDENCE'
                print(f"  -> WARNING: Low confidence detection! ({confidence:.2f} < {CONFIDENCE_THRESHOLD})")

            # Log to console
            print(f"  -> Classification: '{predicted_class}' with {confidence:.2%} confidence.")

            # Prepare data for CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            image_name = os.path.basename(image_path)
            
            # Log to CSV
            csv_writer.writerow([timestamp, image_name, predicted_class, f"{confidence:.4f}", flag])
            
            # Wait for the next "frame"
            time.sleep(SIMULATION_DELAY_SECONDS)

    print("\n--- Simulation Complete ---")
    print(f"Results have been logged to {RESULTS_CSV_PATH}")