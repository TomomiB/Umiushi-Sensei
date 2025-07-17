import os
import numpy as np
import pickle
from PIL import Image, ImageOps
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

### Multi-class Labeling

def resize_and_pad(img, target_size=(224, 224)):
    # Resize image preserving aspect ratio
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    # Calculate padding to reach target size
    delta_w = target_size[0] - img.size[0]
    delta_h = target_size[1] - img.size[1]
    padding = (delta_w // 2, delta_h // 2, 
               delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    # Pad image and return
    new_img = ImageOps.expand(img, padding, fill=(0,0,0))  # black padding
    return new_img

# Function to load and preprocess the images for multi-class classification
def load_and_preprocess_data(data_dir, img_size=(224, 224)):
    images = []
    labels = []
    class_names = os.listdir(data_dir)  # Each folder is a class

    # Create a label encoder to map class names to integer labels
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)  # Fit to the class names in your dataset

    for class_name in class_names:
        class_folder = os.path.join(data_dir, class_name)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = Image.open(img_path).convert('RGB')
                
                # Resize with aspect ratio preserved + pad to square
                img = resize_and_pad(img, target_size=img_size)
                
                img = np.array(img) / 255.0  # Normalize
                images.append(img)
                labels.append(class_name)

    # Convert lists to numpy arrays
    images = np.array(images)
    
    # Encode labels into integers
    labels = label_encoder.transform(labels)
    
    # Convert labels to one-hot encoding (for multi-class classification)
    labels = to_categorical(labels, num_classes=len(class_names))

    return images, labels, label_encoder

# Example usage:
data_dir = '..\images'  # Parent directory containing subfolders of sea slug types
images, labels, label_encoder = load_and_preprocess_data(data_dir)

# save the label encoder
with open("../label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

### Update the model for multi-class classification
# Define the model architecture
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(40, activation='softmax')  # 40 sea slug types
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

### Train the model
model.fit(images, labels, batch_size=32, epochs=10, validation_split=0.2)

### Save the model
print("Saving model to:", os.getcwd())
model.save("../sea_slug_classifier.h5")