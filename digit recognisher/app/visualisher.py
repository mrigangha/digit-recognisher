import tkinter as tk
from tkinter import filedialog
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from PIL import Image 


device = torch.device("cpu")

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
model.load_state_dict(torch.load("../model.pth"))
model.eval()

# Load and preprocess the image



def browse_image():
    selected_image = filedialog.askopenfilename()
    print(selected_image)
    image_path = selected_image # Replace 'image.png' with your image file
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
        t.delete('1.0', tk.END)
        t.insert(tk.END,"Predicted Output  :: "+str(predicted.item()))
    if selected_image:
        # Load the image using OpenCV
        image = cv2.imread(selected_image)

        # Convert the image to RGB (if necessary)
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Create a Tkinter image from the OpenCV image
        tk_image = tk.PhotoImage(data=cv2.imencode('.png', image)[1].tobytes())

        # Display the image in the Tkinter window
        image_label.config(image=tk_image)
        image_label.image = tk_image

# Create the main window
window = tk.Tk()
window.title("Image Visualizer")
window.geometry("250x300")
# Create a label to display the image
image_label = tk.Label(window)
image_label.pack()

t=tk.Text(window,height=10,width=52)
t.pack()

# Create a button to trigger the image selection dialog
browse_button = tk.Button(window, text="Browse", command=browse_image)
browse_button.pack()

# Start the main loop
window.mainloop()