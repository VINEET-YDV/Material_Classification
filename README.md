# Material_Classification
Build a mini classification pipeline that simulates scrap material classification using image data, and deploy it in a simple real-time simulation loop.

1.  **Clone the Repository**
2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Install Dependencies**
    Create a `requirements.txt` file with the content below and run `pip install -r requirements.txt`.
    ```txt
    torch
    torchvision
    numpy
    opencv-python
    onnxruntime
    scikit-learn
    seaborn
    matplotlib
    Pillow
    tqdm
    ```
4.  **Download the Dataset**
    Download the TrashNet dataset and structure it in a folder named `dataset-resized` with subfolders for each class.

## ▶️ How to Use the Project

1.  **Train the Model**: Run your training script/notebook to generate `waste_classifier_resnet18.pth`.
2.  **Convert the Model to ONNX**: Run `converter.py` to generate `waste_classifier.onnx`.
3.  **Run the Simulation**:
    -   Open `conveyor_simulation_video.py`.
    -   Run `python conveyor_simulation_video.py`.
