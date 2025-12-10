import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
import numpy as np

# ================== CONFIG ==================
DATA_DIR = r"D:\gesture_robot\datasets"   # adjust if needed
MODEL_PT = "gesture_model.pt"
MODEL_ONNX = "gesture_model.onnx"
CLASS_NAMES_FILE = "class_names.txt"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-3
SEED = 42
# ============================================


def build_dataloaders():
    """Create train/val dataloaders with stratified split."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR)
    targets = [full_dataset[i][1] for i in range(len(full_dataset))]

    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=targets,
        random_state=SEED
    )

    class TransformSubset:
        def __init__(self, subset, transform, classes, class_to_idx):
            self.subset = subset
            self.transform = transform
            self.classes = classes
            self.class_to_idx = class_to_idx

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.subset)

    train_set = TransformSubset(Subset(full_dataset, train_idx), transform_train,
                                full_dataset.classes, full_dataset.class_to_idx)
    val_set = TransformSubset(Subset(full_dataset, val_idx), transform_val,
                              full_dataset.classes, full_dataset.class_to_idx)

    # Save class names
    with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
        for name in full_dataset.classes:
            f.write(name + "\n")

    # num_workers=0 for Windows safety
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)
    return train_loader, val_loader, full_dataset.classes


def build_model(num_classes: int):
    """EfficientNet-B0 with frozen backbone and custom head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


def export_onnx(model, class_names, device):
    """Export the trained PyTorch model to ONNX format."""
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy,
        MODEL_ONNX,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX exported to {MODEL_ONNX} (classes: {class_names})")


def main():
    if not Path(DATA_DIR).is_dir():
        raise FileNotFoundError(f"{DATA_DIR} does not exist")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = build_dataloaders()
    model = build_model(len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_acc = 0.0
    patience = 0
    max_patience = 5

    for epoch in range(EPOCHS):
        model.train()
        run_loss = 0.0
        run_correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            run_loss += loss.item() * images.size(0)
            run_correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss = run_loss / total
        train_acc = run_correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"- train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"- val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PT)
            patience = 0
            print(f"  ✓ Saved best checkpoint ({MODEL_PT}), val acc {val_acc:.3f}")
        else:
            patience += 1
            if patience >= max_patience:
                print("Early stopping")
                break

    # Load best checkpoint for export
    model.load_state_dict(torch.load(MODEL_PT, map_location=device))
    export_onnx(model.to(device), class_names, device)
    print("Done.")


if __name__ == "__main__":
    main()

