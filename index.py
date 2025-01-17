import cv2
import numpy as np
from keras.models import load_model
import threading
import time
import json
import os

def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):
    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x0, y0 + baseline), (max(xt, x0 + w), yt), color, 2)
    cv2.rectangle(img, (x0, y0 - h), (x0 + w, y0 + baseline), color, -1)
    cv2.putText(img, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    return img

# Load Haar Cascade and Keras model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model_cnn/model_cnn_face_recognition-test2.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model_cnn/model_cnn_face_recognition-gray.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model_cnn/model_cnn_gray2.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model_cnn/model_cnn_gray3.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model/model_new_gray_api.keras')
model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model/model_new_gray_api_test.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model/model_new_rgb_api.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model_cnn/model_cnn_face_recognition-test3.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model_cnn/model_cnn_face_recognition-testAPI.keras')
# model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model_cnn/model_cnn_face_recognition-testAPIWeb.keras')

print("[INFO] Model successfully loaded...")

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

# Global variables for threading
frame = None
label_text = "Detecting..."
confidence = 0
skip_frames = 5  # Skip prediction every few frames for efficiency
frame_count = 0

# RGB color channels
# def predict_face(face_img):
#     global label_text, confidence
#     face_img = cv2.resize(face_img, (50, 50))
    
#     # Mengonversi grayscale menjadi RGB
#     if len(face_img.shape) == 2:  # Memeriksa jika gambar berformat grayscale
#         face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)  # Mengonversi ke RGB
    
#     face_img = face_img.reshape(1, 50, 50, 3)  # Merubah bentuk untuk model
    
#     result = model.predict(face_img)
#     idx = result.argmax(axis=1)[0]
#     confidence = result.max(axis=1)[0] * 100
    
#     if 0 <= idx < len(labels):
#         label_text = "%s (%.2f %%)" % (labels[idx], confidence)
#     else:
#         label_text = "Unknown"

# grayscale channel
def predict_face(face_img):
    global label_text, confidence
    face_img = cv2.resize(face_img, (50, 50))
    
    # Pastikan input dalam format (50, 50, 1) untuk grayscale
    if len(face_img.shape) == 2:  # Grayscale
        face_img = face_img[:, :, np.newaxis]  # Tambahkan dimensi saluran untuk CNN
    
    face_img = face_img.reshape(1, 50, 50, 1)  # Bentuk input ke model
    
    result = model.predict(face_img)
    idx = result.argmax(axis=1)[0]
    confidence = result.max(axis=1)[0] * 100
    
    if 0 <= idx < len(labels):
        label_text = "%s (%.2f %%)" % (labels[idx], confidence)
    else:
        label_text = "Unknown"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, original_frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(original_frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        
        # Run prediction only on every `skip_frames`
        if frame_count % skip_frames == 0:
            prediction_thread = threading.Thread(target=predict_face, args=(face_img,))
            prediction_thread.start()
        
        # Draw the prediction result
        frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0, 255, 255), text_color=(50, 50, 50))

    frame_count += 1  # Increment frame count
    cv2.imshow('Detect Face', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()