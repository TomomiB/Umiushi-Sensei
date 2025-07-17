import os
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

### Load the model and label encoder
model_path = "../train/sea_slug_classifier.h5"
label_encoder_path = "../train/label_encoder.pkl"

if os.path.exists(model_path):
    model = load_model(model_path)
    print("Loaded existing model.")
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

if os.path.exists(label_encoder_path):
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
else:
    raise FileNotFoundError(f"Label file not found: {label_encoder_path}")

### Inference - Predict Sea Slug Type
# Load a new image for prediction
img_path = '../predict_please/hiroumiushi.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize the image

# Predict the class probabilities
predictions = model.predict(img_array)

# Get the predicted class index (highest probability)
predicted_class_idx = np.argmax(predictions)

# Decode the predicted class index to the sea slug name
predicted_class = label_encoder.inverse_transform([predicted_class_idx])

print(f"Predicted Sea Slug: {predicted_class[0]}")