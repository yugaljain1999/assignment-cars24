import os
import torch.nn as nn
import torch
from PIL import Image
from io import BytesIO
from torchvision import transforms
from utils.config import MODEL_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image classifier model
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def load_model():
    """Load saved model"""
    classifier = ImageClassifier().to(device)
    with open(os.path.join(MODEL_DIR,"model_state.pt"), 'rb') as f:
        classifier.load_state_dict(torch.load(f, map_location=device))
    classifier.eval()
    print("Model Loaded")
    return classifier  
       

def read_imagefile(file) -> Image.Image:
    """Read uploaded image file"""
    image = Image.open(BytesIO(file))
    return image

def predict(image: Image.Image):
    """Predict label for input image"""
    global model
    model = load_model()

    image = image.resize((28,28))
    image_transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    output = model(image_tensor)
    predicted_label = torch.argmax(output)
    return predicted_label
