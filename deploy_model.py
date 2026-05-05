import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import PIL.Image as Image
import os

# 1. THE ARCHITECTURE (Must be identical to training)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 4 * 4) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. SETUP DEVICE AND LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = EmotionCNN(num_classes=7)

# FIXED: Added 'r' for raw string to handle Windows backslashes
MODEL_PATH = r'D:\4_second term\AI\Emotion detection project\model_wieghts.pth'

if os.path.exists(MODEL_PATH):
    # Load the weights first
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    
    # CRITICAL: Move model to device AFTER loading weights
    model.to(device)
    model.eval() 
    print("Weights loaded successfully and model moved to GPU!")
else:
    print(f"Error: {MODEL_PATH} not found. Check the file path.")

# 3. PREDICTION FUNCTION
def predict_emotion(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((75, 75)),
        transforms.ToTensor(),
    ])

    # Load image and ensure it's in a format PIL can process easily
    img = Image.open(image_path).convert('RGB') 
    
    # Transform and move the tensor to the SAME device as the model
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    return classes[pred.item()]

# 4. TEST IT
if __name__ == "__main__":
    # FIXED: Ensured this path is also a raw string
    test_image = r"D:\4_second term\AI\Emotion detection project\ggggggggg.jpeg"
    
    if os.path.exists(test_image):
        result = predict_emotion(test_image)
        print("-" * 30)
        print(f"IMAGE: {os.path.basename(test_image)}")
        print(f"RESULT: {result.upper()}")
        print("-" * 30)
    else:
        print(f"Error: Could not find the test image at {test_image}")