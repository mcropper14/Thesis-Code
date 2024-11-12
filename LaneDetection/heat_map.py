
"""
Generate hapmap for the model

"""

import efficient as driving_model
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self):
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().numpy(), 0)  
        heatmap /= np.max(heatmap)  
        return heatmap

    def visualize(self, input_image, heatmap):
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize(input_image.size, Image.BILINEAR)
        heatmap = np.array(heatmap)

        superimposed_img = np.array(input_image) * 0.6 + heatmap[:, :, np.newaxis] * 0.4
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        plt.figure(figsize=(8, 8))
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.savefig('heatmap_visualization2.png')  #save for paper
        print("Heatmap saved as heatmap_visualization.png")


model = driving_model.DrivingEfficientNetLSTM()
#change to current model path - in form of best_driving_cnn#.pth, pytorch model
model.load_state_dict(torch.load('best_driving_cnn9.pth', weights_only=True)) 
model.eval()

grad_cam = GradCAM(model, model.conv3)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = driving_model.DrivingDataset(csv_file='collected_movement_data/collected_movement_data/movement_data.csv', 
                         root_dir='collected_movement_data', transform=transform)


sample_image, sample_labels = dataset[0]  
sample_image_pil = to_pil_image(sample_image)  
sample_image = sample_image.unsqueeze(0)  


output = model(sample_image)


model.zero_grad()
output[0, 0].backward()  

#generate heatmap 
heatmap = grad_cam.generate_heatmap()
grad_cam.visualize(sample_image_pil, heatmap)
