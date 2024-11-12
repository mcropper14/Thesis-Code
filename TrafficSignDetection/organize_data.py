import os
import shutil
import random

def organize_dataset(base_dir, output_dir, train_ratio=0.8):
    images_path = os.path.join(base_dir, 'images')
    labels_path = os.path.join(base_dir, 'labels')
    all_images = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    all_labels = [f for f in os.listdir(labels_path) if os.path.isfile(os.path.join(labels_path, f))]
    all_images.sort()
    all_labels.sort()

    assert len(all_images) == len(all_labels), "Mismatch between images and labels"

    #training and vali
    data = list(zip(all_images, all_labels))
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]

    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    def copy_files(data, split):
        for img, lbl in data:
            shutil.copy(os.path.join(images_path, img), os.path.join(output_dir, 'images', split, img))
            shutil.copy(os.path.join(labels_path, lbl), os.path.join(output_dir, 'labels', split, lbl))

    copy_files(train_data, 'train')

    copy_files(val_data, 'val')

    print(f"Dataset organized into {output_dir} with {len(train_data)} training and {len(val_data)} validation samples.")


base_dir = "traffic_sign_detection/obstacle_data"
output_dir = 'traffic_sign_detection/obstacle_data_split'
organize_dataset(base_dir, output_dir)
