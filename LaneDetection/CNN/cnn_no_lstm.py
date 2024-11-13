"""
Old CNN without LSTM
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

class DrivingCNN(nn.Module):
    def __init__(self):
        super(DrivingCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 5)
        #input 3, 100, 100
        #output 16 * 96 * 96 (100 - 5 + 1)
        #output 16 * 44 * 44 
        self.conv2 = nn.Conv2d(16, 32, 5)
        #input 16, 48, 48
        #output 32 * 44 * 44 
        
        self.conv3 = nn.Conv2d(32, 64, 5)
        #input 32, 22, 22
        #output 64 * 18 * 18 #9 * 9 
        #ouput 64 * 18 * 18
        
        self.fc1 = nn.Linear(64 * 9 * 9, 500)
        self.fc2 = nn.Linear(500, 2) #speed, turn

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 9 * 9)  #22 * 22 update dim
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 9 * 9) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DrivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 1]) #0
        image = Image.open(img_name).convert("RGB")
        #print(f"Original image size: {image.size}") #test
        #640, 480 
        
        speed = self.annotations.iloc[idx, 2] #linear_x
        turn = self.annotations.iloc[idx, 3] #angular_z
        labels = torch.tensor([speed, turn], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
            #print(f"Transformed image size: {image.size()}") #channels, height, width
            #3, 100, 100

        return image, labels

# Load and transform the dataset
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


patience = 10  
best_accuracy = 0.0
epochs_without_improvement = 0
num_epochs = 100  

for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    model.eval()  
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            total_correct += ((outputs - labels).abs() < 0.1).all(dim=1).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f'Test Accuracy after Epoch {epoch+1}: {accuracy:.4f}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_driving_cnn9.pth')
        print(f'New best model saved with accuracy: {best_accuracy:.4f}')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f'Stopping early at epoch {epoch+1}')
            break

print('Finished Training')
print(f'Best Test Accuracy: {best_accuracy:.4f}')
