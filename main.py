# Import necessary libraries
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

# Install torchvision library if not already installed
!pip install torchvision

# Define a series of image transformations to preprocess the input image
transform = transforms.Compose([
    transforms.Resize(256),        # Resize the image to 256x256 pixels
    transforms.CenterCrop(224),    # Center crop the image to 224x224 pixels
    transforms.ToTensor(),         # Convert the image to a PyTorch tensor
    transforms.Normalize(           # Normalize the image data
        mean=[0.485, 0.456, 0.406],  # Mean values for RGB channels
        std=[0.229, 0.224, 0.225]    # Standard deviation values for RGB channels
    )
])

# Download an example image for classification
!wget https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg -O dog.jpg

# Download the dataset text file containing class labels
!wget https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt

# Open the example image using the PIL library
img = Image.open("dog.jpg")

# Apply the defined transformations to the image
img_t = transform(img)

# Add an extra dimension to the image tensor to create a batch
batch_t = torch.unsqueeze(img_t, 0)

# Load the AlexNet model pre-trained on ImageNet dataset
alexnet = models.alexnet(pretrained=True)

# Print the architecture of the loaded AlexNet model
print(alexnet)

# Set the model to evaluation mode
alexnet.eval()

# Perform forward pass to get the model's output
out = alexnet(batch_t)

# Print the shape of the model's output
print(out.shape)

# Load class labels from the text file
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Sort the output and calculate class probabilities
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# Print the top 5 predicted classes and their probabilities
top5 = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
print(top5)
