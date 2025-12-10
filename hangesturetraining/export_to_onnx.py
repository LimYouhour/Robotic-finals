import torch
from torchvision import models
from torch import nn

MODEL_PATH = "gesture_model.pt"
ONNX_PATH = "gesture_model.onnx"
NUM_CLASSES =  len(open("class_names.txt").read().splitlines())  # loads number of classes
IMG_SIZE = 224

def build_model(num_classes: int):
    """Rebuild the same model architecture used in training."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Recreate the classifier head (same as training)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

# Load model
model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Dummy input (batch=1, 3×224×224)
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)

print(f"Model exported successfully to {ONNX_PATH}")
