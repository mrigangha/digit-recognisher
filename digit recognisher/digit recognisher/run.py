import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from PIL import Image 

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NN() # Replace 'mnist_model.pt' with your model file
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load and preprocess the image
image_path = 'a.png' # Replace 'image.png' with your image file
image = Image.open(image_path).convert('L') # Convert to grayscale
transform = transforms.Compose([
    transforms.Resize((28, 28)), # Resize to MNIST size
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize the image
])
image = transform(image).unsqueeze(0) # Add a batch dimension

# Make predictions
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)
print('Predicted digit:', predicted.item())