import os
import sys
from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
import numpy as np
from flask_cors import CORS
import json
import traceback
import time
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from mtcnn import MTCNN

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://face-recognition-lyart.vercel.app",  # URL produksi Next.js
            "http://localhost:3000"  # URL pengembangan lokal
        ],
        "supports_credentials": False
    }
})

# Constants
DATA_DIR = "public/imagesFace/daftar"
MODEL_DIR = "model"
MODEL_NAME = "model_new_gray_api_test.keras"
JSON_PATH = os.path.abspath("public/nim_labels.json")
MODEL_PREDICT = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model/model_new_gray_api_test.keras')

# Fungsi untuk mencetak langkah-langkah dengan log
def print_step(message):
    print(f"[STEP] {message}")

# simpan gambar
def save_image(file, dir_path, file_name):
    """Helper function to save a file."""
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, file_name)
    file.save(file_path)
    return file_path

# Fungsi untuk mendeteksi dan memotong wajah menggunakan MTCNN (Grayscale)
def detect_face(img):
    detector = MTCNN()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img_rgb)

    if result:
        x, y, width, height = result[0]['box']
        x, y = max(0, x), max(0, y)
        cropped_face = img[y:y + height, x:x + width]
        cropped_face = cv2.resize(cropped_face, (50, 50))
        return cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale
    else:
        return None
    
def detect_face_predict(frame):
    detector = MTCNN()
    result = detector.detect_faces(frame)
    if result:
        x, y, width, height = result[0]['box']
        cropped_face = frame[y:y + height, x:x + width]
        cropped_face_resized = cv2.resize(cropped_face, (50, 50))
        
        if len(cropped_face_resized.shape) == 2:  # Jika gambar grayscale (2 dimensi)
            cropped_face_resized = cropped_face_resized[:, :, np.newaxis]  # Tambahkan dimensi channel
        elif cropped_face_resized.shape[-1] == 3:  # Jika gambar RGB
            cropped_face_resized = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        return cropped_face_resized, (x, y, width, height)  # Kembalikan wajah dan bounding box
    return None, None

# Fungsi untuk augmentasi gambar (Grayscale)
def img_augmentation(img):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    transformations = [
        cv2.getRotationMatrix2D(center, angle, 1.0) for angle in [5, -5, 10, -10]
    ] + [
        np.float32([[1, 0, tx], [0, 1, 0]]) for tx in [3, -3, 6, -6]
    ] + [
        np.float32([[1, 0, 0], [0, 1, ty]]) for ty in [3, -3, 6, -6]
    ]

    augmented_images = [cv2.warpAffine(img, t, (w, h), borderValue=255) for t in transformations]
    intensity_modifications = [cv2.add(img, delta) for delta in [10, 30, -10, -30, 15, 45, -15, -45]]
    return augmented_images + intensity_modifications

# Fungsi untuk proses dataset (preprocessing, augmentasi, dan penyimpanan)
def process_dataset(data_dir):
    print_step("Memulai proses load, preprocessing, augmentasi, dan penyimpanan gambar")
    names = []
    images = []

    if not os.path.exists(data_dir):
        print(f"[ERROR] Direktori {data_dir} tidak ditemukan.")
        return [], []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        print_step(f"Memproses folder: {folder}")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        # Periksa jumlah gambar dalam folder
        if len(files) < 10:
            print(f"[WARNING] Folder {folder} dilewati karena kurang dari 10 gambar.")
            continue

        # Ambil antara 10 hingga 30 gambar
        if len(files) > 20:
            files = np.random.choice(files, 20, replace=False)
        elif len(files) >= 10:
            files = np.random.choice(files, len(files), replace=False)

        for filename in files:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                img_resized = cv2.resize(img, (50, 50))
                face = detect_face(img_resized)  # Deteksi wajah setelah resize
                if face is not None:
                    images.append(face)
                    names.append(folder)

                    # Augmentasi
                    augmented_images = img_augmentation(face)
                    for j, aug_img in enumerate(augmented_images):
                        if aug_img.shape == (50, 50):  # Validasi dimensi hasil augmentasi
                            images.append(aug_img)
                            names.append(folder)

    print_step("Proses preprocessing dan augmentasi selesai")
    return images, names

# Fungsi untuk menyeimbangkan data
def balance_data(names, images, n=200):
    print_step("Memulai penyeimbangan data")
    
    # Hitung jumlah data awal per kelas
    unique_labels, counts_before = np.unique(names, return_counts=True)
    print("[INFO] Jumlah data sebelum balancing:")
    for label, count in zip(unique_labels, counts_before):
        print(f"  - {label}: {count} sampel")

    balanced_names = []
    balanced_images = []

    for label in unique_labels:
        label_indices = np.where(np.array(names) == label)[0]
        selected_indices = np.random.choice(label_indices, min(n, len(label_indices)), replace=False)

        balanced_names.extend([names[i] for i in selected_indices])
        balanced_images.extend([images[i] for i in selected_indices])

    # Hitung jumlah data setelah balancing
    unique_labels, counts_after = np.unique(balanced_names, return_counts=True)
    print("[INFO] Jumlah data setelah balancing:")
    for label, count in zip(unique_labels, counts_after):
        print(f"  - {label}: {count} sampel")

    print(f"[INFO] Data berhasil diseimbangkan. Jumlah sampel maksimum per kelas: {n}")
    return balanced_names, balanced_images

# Labels Loader
def load_labels(JSON_PATH):
    if not os.path.exists(JSON_PATH):
        print_step(f"[ERROR] File JSON tidak ditemukan: {JSON_PATH}")
        return []
    try:
        with open(JSON_PATH, "r") as file:
            labels = json.load(file)
            print_step(f"[INFO] Label berhasil dimuat: {labels}")
            return labels
    except json.JSONDecodeError as e:
        print_step(f"[ERROR] Gagal memuat file JSON: {e}")
        return []

labels = load_labels(JSON_PATH)

# Fungsi untuk membuat CNN
def cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu", input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding="valid", activation="relu"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@app.route('/', methods=['GET'])
def home():
    return 'hallo sobat'
from flask import Response, stream_with_context

@app.route('/training', methods=['POST'])
def train_model():
    def generate_progress():
        steps = [
            "Proses pelatihan dimulai...",
            "Loading dan preprocessing dataset...",
            "Encoding label...",
            "Training model...",
            "Evaluasi model...",
            "Pelatihan selesai!"
        ]
        for step in steps:
            yield f"[STEP] {step}\n"
            sys.stdout.flush()  
            time.sleep(2)  # Simulasi jeda antara setiap langkah

    return Response(stream_with_context(generate_progress()), mimetype='text/plain')


if __name__ == "__main__":
    app.run(debug=False)

