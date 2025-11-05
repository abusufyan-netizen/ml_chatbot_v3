"""
Predict Handwritten Digit using Trained CNN Model
-------------------------------------------------
Author: Your Name
Date: November 2025

Description:
------------
This script loads the trained CNN model (mnist_cnn_best.pth)
and predicts the handwritten digit from a custom image file.

Usage:
------
python predict_image.py --image your_image.png
"""

# ==============================
# 📦 IMPORT LIBRARIES
# ==============================
import torch
from torchvision import transforms
from PIL import Image
import argparse
from train_model import CNNModel, DEVICE, MODEL_PATH

# ==============================
# 🧠 IMAGE PREDICTION FUNCTION
# ==============================
def predict_image(image_path):
    """Predict a digit from a given image"""
    # Load model
    model = CNNModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Define transform (same as training)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True).item()

    print(f"\n🖼️ Image Path: {image_path}")
    print(f"✅ Predicted Digit: {pred}\n")
    return pred

# ==============================
# 🚀 MAIN ENTRY POINT
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a handwritten digit using trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (PNG/JPG)")
    args = parser.parse_args()

    predict_image(args.image)
