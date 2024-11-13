
"""
Lane Detection CNN Model
EfficientNet with LSTM

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import pandas as pd
import os
import torch.onnx

class DrivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, sequence_length=3, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.annotations) - self.sequence_length + 1

    def __getitem__(self, idx):
        images = []
        for i in range(self.sequence_length):
            img_name = os.path.join(self.root_dir, self.annotations.iloc[idx + i, 1])
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        images = torch.stack(images)  #(sequence_length, C, H, W)
        
        speed = self.annotations.iloc[idx + self.sequence_length - 1, 2]  #speed at last frame
        turn = self.annotations.iloc[idx + self.sequence_length - 1, 3]   #turn at last frame
        labels = torch.tensor([speed, turn], dtype=torch.float32)

        return images, labels

class DrivingEfficientNetLSTM(nn.Module):
    def __init__(self, sequence_length=3, hidden_size=128, num_layers=1):
        super(DrivingEfficientNetLSTM, self).__init__()
        
        efficientnet = efficientnet_b0(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-2])
        self.feature_dim = efficientnet.classifier[1].in_features #1280
        
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, 2) #speed and turn   

    def forward(self, x):
        batch_size, seq_length, C, H, W = x.size()
        
        #(batch_size * seq_length, C, H, W) for feature extraction
        x = x.view(batch_size * seq_length, C, H, W)  
        x = self.feature_extractor(x)                
        x = x.mean([2, 3]) 
        x = x.view(batch_size, seq_length, -1)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :] 
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


sequence_length = 3  #5
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


dataset = DrivingDataset(csv_file='collected_movement_data/collected_movement_data/movement_data.csv', 
                         root_dir='collected_movement_data', 
                         sequence_length=sequence_length, 
                         transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = DrivingEfficientNetLSTM(sequence_length=sequence_length)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
patience = 10
best_accuracy = 0.0
epochs_without_improvement = 0
total_train_correct = 0
total_train_samples = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_train_correct = 0
    total_train_samples = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        #train
        total_train_correct += ((outputs - labels).abs() < 0.1).all(dim=1).sum().item()
        total_train_samples += labels.size(0)
    
    train_accuracy = total_train_correct / total_train_samples
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}')
    
    #test
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            total_correct += ((outputs - labels).abs() < 0.1).all(dim=1).sum().item()
            total_samples += labels.size(0)
    
    test_accuracy = total_correct / total_samples
    print(f'Test Accuracy after Epoch {epoch+1}: {test_accuracy:.4f}')
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_driving_efficientnet_lstm.pth')
        print(f'New best model saved with accuracy: {best_accuracy:.4f}')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f'Stopping early at epoch {epoch+1}')
            break

print('Finished Training')
print(f'Best Test Accuracy: {best_accuracy:.4f}')



dummy_input = torch.randn(1, sequence_length, 3, 224, 224)  # (batch_size, sequence_length, C, H, W)
onnx_file_path = 'driving_efficientnet_lstm.onnx'
torch.onnx.export(
    model,                     
    dummy_input,              
    onnx_file_path,            
    export_params=True,      
    opset_version=11,         
    do_constant_folding=True, 
    input_names=['input'],     
    output_names=['output'],   
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Model has been saved in ONNX format at '{onnx_file_path}'")
