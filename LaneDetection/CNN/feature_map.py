"""
Visualizes the feature maps from each convolutional layer for a single image.
from *.pth (not .onnx) model file
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

class DrivingCNN(nn.Module):
    def __init__(self):
        super(DrivingCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 9 * 9, 500)
        self.fc2 = nn.Linear(500, 2)  

    def forward(self, x):
        feature_maps = {}  

        x = F.relu(self.conv1(x))
        feature_maps['conv1'] = x

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        feature_maps['conv2'] = x

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))
        feature_maps['conv3'] = x

        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 64 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x, feature_maps

class DrivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 1])
        image = Image.open(img_name).convert("RGB")
        
        speed = self.annotations.iloc[idx, 2]
        turn = self.annotations.iloc[idx, 3]
        labels = torch.tensor([speed, turn], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

def visualize_feature_maps(feature_maps):
    for layer_name, feature_map in feature_maps.items():
        num_filters = feature_map.shape[1]  # Number of filters
        fig, axes = plt.subplots(1, min(num_filters, 6), figsize=(15, 5))
        fig.suptitle(f'Feature Maps from {layer_name}', fontsize=16)

        for i in range(min(num_filters, 6)):
            ax = axes[i]
            ax.imshow(feature_map[0, i].detach().cpu().numpy(), cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filter {i+1}')
        plt.show()
        plt.savefig("feature_map.png")

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

dataset = DrivingDataset(csv_file='collected_movement_data/collected_movement_data/movement_data.csv', 
                         root_dir='collected_movement_data', transform=transform)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = DrivingCNN()
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)


image, _ = dataset[0]  
image = image.unsqueeze(0) 
_, feature_maps = model(image)
visualize_feature_maps(feature_maps)
