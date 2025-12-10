"""
Export an existing trained PyTorch checkpoint to ONNX without retraining.
Requires: gesture_model.pt and class_names.txt in the same folder.
Usage: python export_only_onnx.py
"""
import os
import torch
from train_gesture_robot import build_model, IMG_SIZE

MODEL_PT = "gesture_model.pt"
MODEL_ONNX = "gesture_model.onnx"
CLASS_NAMES_FILE = "class_names.txt"


def load_class_names(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Class names file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_class_names(CLASS_NAMES_FILE)
    if not os.path.exists(MODEL_PT):
        raise FileNotFoundError(f"Model file not found: {MODEL_PT}")

    print(f"Loading model from {MODEL_PT} ...")
    model = build_model(len(class_names))
    state_dict = torch.load(MODEL_PT, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    print(f"Exporting to {MODEL_ONNX} ...")
    torch.onnx.export(
        model,
        dummy,
        MODEL_ONNX,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Done. ONNX saved to {MODEL_ONNX}")


if __name__ == "__main__":
    main()

