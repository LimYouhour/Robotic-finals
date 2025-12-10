import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = r"D:\gesture_robot\datasets"
MODEL_PATH = "gesture_model.pt"
CLASS_NAMES_FILE = "class_names.txt"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-3
SEED = 42


def build_dataloaders():
    """Build dataloaders with stratified split to prevent class imbalance."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

    # Load full dataset without transforms first
    full_dataset = datasets.ImageFolder(DATA_DIR)
    
    # Get indices and labels for stratified split
    targets = [full_dataset[i][1] for i in range(len(full_dataset))]
    
    # Stratified split ensures balanced class distribution
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=targets,
        random_state=SEED
    )
    
    # Create subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    # Apply transforms by creating wrapper datasets
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
    
    train_set = TransformSubset(train_subset, transform_train, 
                                full_dataset.classes, full_dataset.class_to_idx)
    val_set = TransformSubset(val_subset, transform_val,
                              full_dataset.classes, full_dataset.class_to_idx)
    
    # Print class distribution
    train_labels = [targets[i] for i in train_indices]
    val_labels = [targets[i] for i in val_indices]
    print("\nClass distribution:")
    for i, class_name in enumerate(full_dataset.classes):
        train_count = train_labels.count(i)
        val_count = val_labels.count(i)
        print(f"  {class_name}: Train={train_count}, Val={val_count}")
    
    with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
        for name in full_dataset.classes:
            f.write(name + "\n")

    # Use num_workers=0 on Windows to avoid multiprocessing issues
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, full_dataset.classes


def build_model(num_classes: int):
    """Build model with increased dropout for better regularization."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze base model
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace classifier with higher dropout
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


def compute_confusion_matrix(model, loader, class_names, device, out_path="confusion_matrix.png"):
    """Compute and save confusion matrix on the provided loader."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


def train():
    if not Path(DATA_DIR).is_dir():
        raise FileNotFoundError(f"{DATA_DIR} does not exist")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    train_loader, val_loader, class_names = build_dataloaders()
    model = build_model(len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, 
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += images.size(0)

        train_loss = running_loss / total_samples
        train_acc = running_correct / total_samples

        # Validation
        model.eval()
        val_correct = 0
        val_loss = 0.0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"- Train: loss={train_loss:.4f} acc={train_acc:.3f} "
              f"- Val: loss={val_loss:.4f} acc={val_acc:.3f}")

        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved checkpoint (val acc: {val_acc:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.3f}")
    
    # Reload best checkpoint and compute confusion matrix on validation set
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    compute_confusion_matrix(model, val_loader, class_names, device)


if __name__ == "__main__":
    train()