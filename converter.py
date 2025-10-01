import torch
import torch.nn as nn
from torchvision import models

# --- 1. Load your trained PyTorch model (same as above) ---
num_classes = 6
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
# After (The fix)
model.load_state_dict(torch.load('waste_classifier_resnet18.pth', map_location=torch.device('cpu')))
model.eval()

# --- 2. Convert to ONNX ---
dummy_input = torch.randn(1, 3, 224, 224)
onnx_program = torch.onnx.export(
    model,
    dummy_input,
    "waste_classifier.onnx",
    export_params=True,
    opset_version=11, # A commonly used version
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
)
print("Model converted to ONNX and saved as waste_classifier.onnx")