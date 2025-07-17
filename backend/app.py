import os
from io import BytesIO
from flask import Flask, send_from_directory, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

app = Flask(__name__, static_folder='../umiushi-sensei/dist', static_url_path='')

# Load model and label encoder
model_path = "sea_slug_classifier.h5"
label_encoder_path = "label_encoder.pkl"

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

# Serve React static files
@app.route('/')
@app.route('/<path:path>')
def serve_react(path=''):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['file']
    img = image.load_img(BytesIO(img_file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_class = label_encoder.inverse_transform([np.argmax(preds)])[0]

    return jsonify({'prediction': pred_class})

if __name__ == '__main__':
    app.run(debug=True)