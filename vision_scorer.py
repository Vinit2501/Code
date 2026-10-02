import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os


def train_production_vision():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Standard ResNet preprocessing & light data augmentation
    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    data_dir = "laptop_images"
    if not os.path.exists(data_dir):
        print(f"ERROR: '{data_dir}' folder not found.")
        return

    full_dataset = datasets.ImageFolder(data_dir, data_transforms)
    classes = full_dataset.classes
    print(f"Detected classes: {classes}")

    # 80/20 Train/Validation Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Optimized batch size for 80-100 total images
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(
        f"Dataset summary: {train_size} training images, {val_size} validation images."
    )

    # Load ResNet18 and adapt output layer for binary classification
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 5
    print("\nStarting Training Phase...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct.double() / total) * 100
        print(
            f"Epoch {epoch + 1}/{epochs} - Training Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%"
        )

    # Export production weights
    torch.save(model.state_dict(), "vision_weights.pth")
    print("\nSUCCESS: Model retrained and saved to 'vision_weights.pth'.")


if __name__ == "__main__":
    train_production_vision()
