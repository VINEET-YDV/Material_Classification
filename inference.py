import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# --- 1. Configuration ---
ONNX_PATH = 'waste_classifier.onnx'
IMAGE_PATH = r'C:\Users\admin\OneDrive\Desktop\Alfastack\recycle-papers-from-boxes.jpg'
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# --- 2. Image Preprocessing ---
# These transformations MUST be the same as the ones used during training.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. Load the ONNX Model ---
try:
    ort_session = ort.InferenceSession(ONNX_PATH)
    print(f"Successfully loaded ONNX model from {ONNX_PATH}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    exit()

# --- 4. Define the Softmax Function ---
# The model outputs raw scores (logits). Softmax converts them to probabilities.
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# --- 5. Run Prediction ---
def predict(image_path):
    """
    Takes an image path, runs inference, and returns the predicted class and confidence.
    """
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None, None
        
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # Create a batch of 1

    # Convert the PyTorch tensor to a NumPy array for ONNX Runtime
    ort_inputs = {ort_session.get_inputs()[0].name: input_batch.numpy()}

    # Run inference
    ort_outs = ort_session.run(None, ort_inputs)
    
    # The output is a list containing one numpy array with the logits
    scores = ort_outs[0][0]
    
    # Apply softmax to get probabilities
    probabilities = softmax(scores)
    
    # Get the top prediction
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx]
    predicted_class = CLASS_NAMES[predicted_idx]

    return predicted_class, confidence

# --- Main execution block ---
if __name__ == "__main__":
    predicted_class, confidence = predict(IMAGE_PATH)
    
    if predicted_class and confidence:
        print(f"\nPredicted Class: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")