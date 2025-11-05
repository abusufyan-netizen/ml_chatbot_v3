import torch
from torchvision import transforms
from PIL import Image
from train_model import CNNModel, DEVICE, MODEL_PATH

# Load model
model = CNNModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Transform input image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load and predict single image
image = Image.open("digit.png")  # Replace with your test image path
image = transform(image).unsqueeze(0).to(DEVICE)
output = model(image)
pred = output.argmax(dim=1, keepdim=True)
print(f"Predicted Digit: {pred.item()}")
