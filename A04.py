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
	return ["CNN_ALPHA", "CNN_BRAVO", "CNN_CHARLIE", "CNN_DELTA", "CNN_ECHO"]

# Provide descriptions for each approach
def get_approach_description(approach_name):
	if approach_name == "CNN_ALPHA":
		return "A simple CNN with 3 convolutional layers and ReLU activations."
	elif approach_name == "CNN_BRAVO":
		return "A CNN with skip connections added."
	elif approach_name == "CNN_CHARLIE":
		return "CNN_ALPHA but with data augmentation."
	elif approach_name == "CNN_DELTA":
		return "CNN_BRAVO but with data augmentation."
	elif approach_name == "CNN_ECHO":
		return "4 convolutional layers with 64 batch size and data augmentation"
	else:
		raise ValueError(f"Unknown approach_name: {approach_name}")

# Define transformations
def get_data_transform(approach_name, training):
	data_transform = v2.Compose([v2.Resize((128,128)), v2.ToPureTensor(), v2.ConvertImageDtype()])
	if training:
		return data_transform
	elif approach_name == (("CNN_CHARLIE" or "CNN_DELTA") or ("CNN_ECHO")):
		data_transform = v2.Compose([v2.Resize((128,128)), v2.ToPureTensor(), v2.ConvertImageDtype(), v2.RandomHorizontalFlip(.5), v2.RandomGrayscale(.25), v2.RandomInvert(.25)])
		return data_transform
	else:
		return data_transform
	
# Set batch size
def get_batch_size(approach_name):
	if approach_name == "CNN_ECHO":
		return 64
	else:
		return 32

# Create models
def create_model(approach_name, class_cnt):
	if approach_name == "CNN_ALPHA":
		return CNN_ALPHA(class_cnt)
	elif approach_name == "CNN_BRAVO":
		return CNN_BRAVO(class_cnt)
	elif approach_name == "CNN_CHARLIE":
		return CNN_CHARLIE(class_cnt)
	elif approach_name == "CNN_DELTA":
		return CNN_DELTA(class_cnt)
	elif approach_name == "CNN_ECHO":
		return CNN_ECHO(class_cnt)
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
	for epoch in range(75):  #Number of epochs
		model.train()

		for batch in train_dataloader:
			if len(batch) == 3: 
				inputs, _, labels = batch
			else: 
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
		batch_size, num_frames, channels, height, width = x.size()
		x = x.view(batch_size * num_frames, channels, height, width)  
		x = self.pool(self.relu(self.conv1(x)))
		x = self.pool(self.relu(self.conv2(x)))
		x = self.pool(self.relu(self.conv3(x)))
		x = x.reshape(batch_size, num_frames, -1)
		x = x.mean(dim=1)  
		x = self.fc(x)
		return x

# Define CNN_BRAVO
class CNN_BRAVO(nn.Module):
	def __init__(self, class_cnt):
		super(CNN_BRAVO, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.skip = nn.Conv2d(16, 64, kernel_size=1, stride=2)
		self.fc = nn.Linear(64 * 16 * 16, class_cnt)
		self.pool = nn.MaxPool2d(2, 2)
		self.relu = nn.ReLU()

	def forward(self, x):
		batch_size, num_frames, channels, height, width = x.size()
		x = x.view(batch_size * num_frames, channels, height, width)

		x1 = self.pool(self.relu(self.conv1(x)))
		x2 = self.pool(self.relu(self.conv2(x1)))
		x3 = self.pool(self.relu(self.conv3(x2) + self.skip(x1)))

		x3 = x3.reshape(batch_size, num_frames, -1)
		x3 = x3.mean(dim=1)

		x3 = self.fc(x3)
		return x3
	

# Define CNN_CHARLIE
class CNN_CHARLIE(nn.Module):
	def __init__(self, class_cnt):
		super(CNN_CHARLIE, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.fc = nn.Linear(64 * 16 * 16, class_cnt)
		self.pool = nn.MaxPool2d(2, 2)
		self.relu = nn.ReLU()

	def forward(self, x):
		batch_size, num_frames, channels, height, width = x.size()
		x = x.view(batch_size * num_frames, channels, height, width)
		x = self.pool(self.relu(self.conv1(x)))
		x = self.pool(self.relu(self.conv2(x)))
		x = self.pool(self.relu(self.conv3(x)))
		x = x.reshape(batch_size, num_frames, -1) 
		x = x.mean(dim=1)
		x = self.fc(x)
		return x

# Define CNN_DELTA
class CNN_DELTA(nn.Module):
	def __init__(self, class_cnt):
		super(CNN_DELTA, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.skip = nn.Conv2d(16, 64, kernel_size=1, stride=2)
		self.fc = nn.Linear(64 * 16 * 16, class_cnt)
		self.pool = nn.MaxPool2d(2, 2)
		self.relu = nn.ReLU()

	def forward(self, x):
		batch_size, num_frames, channels, height, width = x.size()
		x = x.view(batch_size * num_frames, channels, height, width)
		x1 = self.pool(self.relu(self.conv1(x)))
		x2 = self.pool(self.relu(self.conv2(x1)))
		x3 = self.pool(self.relu(self.conv3(x2) + self.skip(x1)))
		x3 = x3.reshape(batch_size, num_frames, -1) 
		x3 = x3.mean(dim=1)
		x3 = self.fc(x3)
		return x3
	
class CNN_ECHO(nn.Module):
    def __init__(self, class_cnt):
        super(CNN_ECHO, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256 * 8 * 8, 512) 
        self.fc2 = nn.Linear(512, class_cnt) 
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)  
        x = self.pool(self.relu(self.conv1(x))) 
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        _, channels, height, width = x.size()
        x = x.view(batch_size, num_frames, channels, height, width) 
        x = x.mean(dim=1) 
        x = x.view(batch_size, -1) 
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x) 
        return x