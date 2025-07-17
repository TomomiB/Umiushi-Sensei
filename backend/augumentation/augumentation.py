import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

# Settings
target_per_class = 70
image_size = (224, 224)
data_dir = '../images'

# Define augmentation generator
augmentor = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through each class
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    image_files = [
    f for f in os.listdir(class_path)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and not f.startswith('aug_')]
    num_existing = len(image_files)

    print(f"{class_name}: {num_existing} images")

    if num_existing >= target_per_class:
        continue  # Skip classes that already meet the target

    # Load original images into memory
    images = []
    for file in image_files:
        img_path = os.path.join(class_path, file)
        img = Image.open(img_path).convert('RGB').resize(image_size)
        images.append(np.array(img))

    images = np.array(images)

    # Start generating new images
    num_to_generate = target_per_class - num_existing
    generated = 0
    batch_size = 10

    print(f"Augmenting {num_to_generate} new images for class '{class_name}'...")

    while generated < num_to_generate:
        batch = augmentor.flow(images, batch_size=batch_size, shuffle=True)
        augmented_images = next(batch)

        for img_array in augmented_images:
            if generated >= num_to_generate:
                break

            img = Image.fromarray((img_array * 255).astype(np.uint8))
            new_filename = f"aug_{generated + 1}.jpg"
            save_path = os.path.join(class_path, new_filename)
            img.save(save_path, format="JPEG")
            generated += 1

    print(f"✅ Done augmenting class '{class_name}' to {target_per_class} images.\n")