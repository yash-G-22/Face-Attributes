from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import cv2
import traceback
from pathlib import Path
import requests

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

os.makedirs('models', exist_ok=True)
MODEL_LOCAL_PATH = Path("models/trained_Model.h5")
MODEL_URL = os.environ.get("MODEL_URL")  # set this on Render if you host model remotely
K = tf.keras.backend

def custom_mse(y_true, y_pred):
    loss = K.square(y_pred - y_true)
    loss = (K.sum(loss, axis=1)) / 100.0
    return loss

# If model not present locally and MODEL_URL provided, download it at startup
if not MODEL_LOCAL_PATH.exists():
    if MODEL_URL:
        print("Downloading model from MODEL_URL...")
        try:
            MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(MODEL_URL, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(MODEL_LOCAL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print("Model downloaded to", MODEL_LOCAL_PATH)
        except Exception as e:
            print("Model download failed:", e)
    else:
        print("Model file not found and MODEL_URL not set. Server will start without a model.")
        
# Load model
combined_model = None
try:
    combined_model = tf.keras.models.load_model('models/trained_Model.h5',
                                                custom_objects={'custom_mse': custom_mse})
    print("✓ Loaded combined model")
except Exception as e:
    print("✗ Model load failed:", e)

gender_labels = {0: 'Male', 1: 'Female'}
ethnicity_labels = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_cv2(bgr_image):
    """Detect faces in image"""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))
    if len(faces) == 0:
        h, w = gray.shape
        return [(0, 0, w, h)]
    return faces

def extract_faces_from_bgr(bgr_image, faces):
    """Extract face crops from image"""
    arr = []
    for (x, y, w, h) in faces:
        face = bgr_image[y:y+h, x:x+w]
        arr.append({'face': face, 'box': [int(x), int(y), int(w), int(h)]})
    return arr

def preprocess_for_combined(bgr_face):
    """Preprocess face: RGB 64x64x3 normalized to [0,1]"""
    rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (64, 64))
    return resized.astype('float32') / 255.0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict age, gender, ethnicity from uploaded image"""
    if 'image' not in request.files:
        return jsonify({'faces': [], 'error': 'No image provided'}), 200
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'faces': [], 'error': 'No file selected'}), 200

    # Read image
    try:
        data = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'faces': [], 'error': 'Cannot decode image'}), 200
    except Exception as e:
        return jsonify({'faces': [], 'error': f'Error reading image: {e}'}), 200

    # Check model loaded
    if combined_model is None:
        return jsonify({'faces': [], 'error': 'Model not loaded on server'}), 200

    # Detect and extract faces
    faces_boxes = detect_faces_cv2(img_bgr)
    extracted = extract_faces_from_bgr(img_bgr, faces_boxes)
    
    if len(extracted) == 0:
        return jsonify({'faces': []}), 200

    try:
        # Prepare batch: [n_faces, 64, 64, 3]
        batch = np.stack([preprocess_for_combined(item['face']) for item in extracted], axis=0)
        
        # Model returns list: [age_preds, gender_preds, ethnicity_preds]
        preds = combined_model.predict(batch, verbose=0)
        age_preds = preds[0]      # shape (n, 1)
        gender_preds = preds[1]   # shape (n, 1) - sigmoid output [0, 1]
        ethnicity_preds = preds[2]  # shape (n, 5) - softmax probabilities

        results = []
        for i, item in enumerate(extracted):
            # Age: round the predicted value
            age_val = float(age_preds[i].reshape(-1)[0])
            age_out = int(round(age_val))

            # Gender: sigmoid output -> threshold at 0.5
            gender_prob = float(gender_preds[i].reshape(-1)[0])
            gender_label = gender_labels[1] if gender_prob >= 0.5 else gender_labels[0]

            # Ethnicity: argmax of softmax probabilities
            eth_probs = ethnicity_preds[i]
            eth_idx = int(np.argmax(eth_probs))
            eth_label = ethnicity_labels.get(eth_idx, 'Unknown')
            eth_conf = float(np.max(eth_probs))

            res = {
                'box': item['box'],
                'age': {'value': age_out},
                'gender': {'label': gender_label},
                'ethnicity': {'label': eth_label}
            }
            results.append(res)

        return jsonify({'faces': results}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'faces': [], 'error': f'Prediction error: {str(e)}'}), 200

@app.route('/api/health', methods=['GET'])
def health():
    status = 'ready' if combined_model is not None else 'model missing'
    return jsonify({'status': status}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
