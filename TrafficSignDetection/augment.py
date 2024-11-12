import os
import cv2
import random
import albumentations as A
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def augment_image(image, background_images, aug):
 
    background = random.choice(background_images)
    background = cv2.resize(background, (image.shape[1], image.shape[0]))
    augmented = aug(image=image)
    image_aug = augmented['image']

    return image_aug

def augment_dataset(images_folder, backgrounds_folder, output_folder, augmentations):
    os.makedirs(output_folder, exist_ok=True)
    
    images = load_images_from_folder(images_folder)
    background_images = load_images_from_folder(backgrounds_folder)
    
    for idx, image in enumerate(images):
        augmented_image = augment_image(image, background_images, augmentations)
        
        output_image_path = os.path.join(output_folder, f"aug_{idx}.jpg")
        cv2.imwrite(output_image_path, augmented_image)

        print(f"Saved augmented image {output_image_path}")

images_folder = 'traffic_sign_detection/Traffic_Signs/images'  
backgrounds_folder =  'traffic_sign_detection/backgrounds' 
output_folder = 'traffic_sign_detection/augmented_images' 


augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RGBShift(p=0.8),
    A.GaussianBlur(p=0.3),
    A.GaussNoise(p=0.3)
])

augment_dataset(images_folder, backgrounds_folder, output_folder, augmentations)
