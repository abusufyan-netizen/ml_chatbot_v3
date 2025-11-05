"""
Handwritten Digit Recognition using PyTorch (with Visual Output)
---------------------------------------------------------------
Author: Your Name
Date: November 2025

Description:
------------
A Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits (0–9).
After training, the program displays test images with predicted and actual labels.

Dependencies:
-------------
torch, torchvision, matplotlib

Usage:
------
python train_model.py
"""

# ==============================
# 📦 IMPORTING LIBRARIES
# ==============================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# ⚙️ CONFIGURATION
# ==============================
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_PATH = "mnist_cnn_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 🧩 DATA PREPARATION
# ==============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# 🧠 MODEL DEFINITION
# ==============================
class CNNModel(nn.Module):
    """CNN for MNIST digit classification"""
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ==============================
# 🏋️ TRAINING FUNCTION
# ==============================
def train(model, device, train_loader, optimizer, criterion, epoch):
    """Trains CNN for one epoch"""
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch}: Training Loss = {running_loss / len(train_loader):.4f}")

# ==============================
# 🧪 TEST FUNCTION
# ==============================
def test(model, device, test_loader, criterion):
    """Evaluates CNN accuracy"""
    model.eval()
    correct = 0
    test_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.2f}%\n")
    return accuracy

# ==============================
# 🎨 VISUALIZATION FUNCTION
# ==============================
def show_results(model, device, test_loader):
    """Displays sample predictions from test dataset"""
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    output = model(images)
    _, preds = torch.max(output, 1)

    # Show 10 sample predictions
    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        img = images[i].cpu().numpy().squeeze()
        plt.imshow(img, cmap="gray")
        plt.title(f"Pred: {preds[i].item()} | True: {labels[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# ==============================
# 🚀 MAIN EXECUTION
# ==============================
def main():
    print("\n📊 Training Handwritten Digit Recognition Model...\n")
    model = CNNModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, criterion, epoch)
        accuracy = test(model, DEVICE, test_loader, criterion)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Model saved with accuracy: {accuracy:.2f}%\n")

    print(f"Training Complete ✅\nBest Model Accuracy: {best_accuracy:.2f}%")
    print(f"Model saved as: {MODEL_PATH}\n")

    # Load best model and visualize predictions
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    show_results(model, DEVICE, test_loader)

# ==============================
# 🧩 ENTRY POINT
# ==============================
if __name__ == "__main__":
    main(            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.2f}%\n")
    return accuracy

# ==============================
# 🚀 MAIN EXECUTION
# ==============================
def main():
    print("📊 Training Handwritten Digit Recognition Model...\n")
    model = CNNModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, criterion, epoch)
        accuracy = test(model, DEVICE, test_loader, criterion)

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Model saved with accuracy: {accuracy:.2f}%\n")

    print(f"Training complete! Best model accuracy: {best_accuracy:.2f}%")
    print(f"Model saved as '{MODEL_PATH}'")

# ==============================
# 🧩 ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()
