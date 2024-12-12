import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
import cv2
import numpy as np
import os
import sys
from prettytable import PrettyTable
from MemeData import *


# Define approach names
def get_approach_names():
    return ["CNN_ALPHA", "CNN_BRAVO"]

# Provide descriptions for each approach
def get_approach_description(approach_name):
    if approach_name == "CNN_ALPHA":
        return "A simple CNN with 3 convolutional layers and ReLU activations."
    elif approach_name == "CNN_BRAVO":
        return "A CNN with skip connections added."
    else:
        raise ValueError(f"Unknown approach_name: {approach_name}")

# Define dataset transformations
def get_data_transform(approach_name, training):
    data_transform = v2.Compose([v2.Resize((128,128)), v2.ToPureTensor(), v2.ConvertImageDtype()])
    #data_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    #data_transform = v2.Compose([v2.Resize((128, 128)),v2.ToTensor(),v2.Normalize(mean=[0.5], std=[0.5])])
    if training:
        return data_transform
    else:
        return data_transform
    
# Set batch size
def get_batch_size(approach_name):
    return 32

# Create model architectu7
def create_model(approach_name, class_cnt):
    if approach_name == "CNN_ALPHA":
        return CNN_ALPHA(class_cnt)
    elif approach_name == "CNN_BRAVO":
        return CNN_BRAVO(class_cnt)
    else:
        raise ValueError(f"Unknown approach_name: {approach_name}")


# Train model
def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    for batch in train_dataloader:
        print(type(batch), len(batch), [b.shape for b in batch if isinstance(b, torch.Tensor)])
        break
    for epoch in range(20):  #Number of epochs
        model.train()

        for batch in train_dataloader:
            if len(batch) == 3:  # Some datasets may provide (inputs, metadata, labels)
                inputs, _, labels = batch
            else:  # Standard case with (inputs, labels)
                inputs, labels = batch
            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model

# Define CNN_ALPHA
class CNN_ALPHA(nn.Module):
    def __init__(self, class_cnt):
        super(CNN_ALPHA, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 16 * 16, class_cnt)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, num_frames, channels, height, width]
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)  # Flatten temporal dimension

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.reshape(batch_size, num_frames, -1)  # Restore temporal dimension
        x = x.mean(dim=1)  # Aggregate over frames (e.g., average pooling)
        
        x = self.fc(x)
        return x

# Define CNN_BRAVO
class CNN_BRAVO(nn.Module):
    def __init__(self, class_cnt):
        super(CNN_BRAVO, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(16, 64, kernel_size=1, stride=2)  # Skip connection
        self.fc = nn.Linear(64 * 16 * 16, class_cnt)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, num_frames, channels, height, width]
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)  # Flatten temporal dimension

        x1 = self.pool(self.relu(self.conv1(x)))
        x2 = self.pool(self.relu(self.conv2(x1)))
        x3 = self.pool(self.relu(self.conv3(x2) + self.skip(x1)))  # Add skip connection

        x3 = x3.reshape(batch_size, num_frames, -1)  # Restore temporal structure
        x3 = x3.mean(dim=1)  # Aggregate over frames (e.g., average pooling)

        x3 = self.fc(x3)
        return x3
