from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import cv2
from mtcnn import MTCNN
from flask_cors import CORS
import json
import os

app = Flask(__name__)
# Konfigurasi CORS untuk menerima kredensial dari localhost dan Ngrok
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://30ca-118-99-65-193.ngrok-free.app"], "supports_credentials":False}})


# Load the Keras model
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model/model-cnn-facerecognition-5.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model/model-cnn-facerecognition-7.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model/model-cnn-facerecognition-9.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model/modelcnn-facerecognition-1.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model_cnn/model_cnn_face_recognition.keras')
model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model_cnn/model_cnn_face_recognition-testAPIWeb.keras')


# Initialize MTCNN for face detection
detector = MTCNN()
def detect_face(frame):
    result = detector.detect_faces(frame)
    if result:
        x, y, width, height = result[0]['box']
        cropped_face = frame[y:y + height, x:x + width]
        cropped_face_resized = cv2.resize(cropped_face, (50, 50))
        
        # Pastikan gambar memiliki 3 saluran
        if cropped_face_resized.shape[-1] != 3:  # Jika bukan RGB
            cropped_face_resized = cv2.cvtColor(cropped_face_resized, cv2.COLOR_GRAY2RGB)
        
        return cropped_face_resized, (x, y, width, height)  # Kembalikan wajah dan bounding box
    return None, None

# labels = [
#     '2118035','2118036','2118100','2118101','2118108','2118112','2118117',
#     '2118120','2118126','2118131','Angelina Jolie', 'Brad Pitt', 'Denzel Washington',
#     'Hugh Jackman','Jennifer Lawrence', 'Johnny Depp', 'Kate Winslet', 'Leonardo DiCaprio', 
#     'Megan Fox', 'Natalie Portman', 'Nicole Kidman', 'Robert Downey Jr',
#     'Sandra Bullock', 'Scarlett Johansson', 'Tom Cruise', 'Tom Hanks',
#     'Will Smith',
# ]

# Path ke file JSON relatif terhadap folder python-face
json_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),  # Lokasi file Python ini
    "../front-facerecognition/public/nim_labels.json"
))

# Fungsi untuk membaca label dari file JSON
def load_labels(json_path):
    if not os.path.exists(json_path):
        print(f"[ERROR] File JSON tidak ditemukan: {json_path}")
        return []

    try:
        with open(json_path, "r") as file:
            labels = json.load(file)
            print(f"[INFO] Label berhasil dimuat: {labels}")
            return labels
    except json.JSONDecodeError as e:
        print(f"[ERROR] Gagal memuat file JSON: {e}")
        return []

# Memuat label dari file JSON
labels = load_labels(json_path)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image'].read()  # ambil gambar dari request
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # Decode image in RGB format

    face, box = detect_face(img)
    if face is None:
        return jsonify({'prediction': 'Face not detected'})

    # Prepare wajah untuk model prediction (already grayscale)
    face = face.reshape(1, 50, 50, 3)  # 1 channel for grayscale

    prediction = model.predict(face)
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))
    # Tentukan nama prediksi
    if confidence < 0.5:  # Jika confidence kurang dari 50%, label "tidak dikenali"
        predicted_name = "tidak dikenali"
    else:
        predicted_name = labels[class_idx]

    return jsonify({
        'prediction': predicted_name, 
        'confidence': confidence, 
        'box': box  # Kembalikan koordinat bounding box
    })

if __name__ == '__main__':
    app.run(debug=True)
