from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
import numpy as np
import os
import cv2
from mtcnn import MTCNN
import time
import tensorflow as tf
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = Flask(__name__)

# Path dataset
dataset_path = "D:/Kuliah/skripsi/face-recognition/python-face/archive/"
logging.info(f"Dataset path: {dataset_path}")

# Path model sebelumnya
model_path = "D:/Kuliah/skripsi/face-recognition/python-face/model/model-cnn-facerecognition-9.keras"
logging.info(f"Model path: {model_path}")

# Path untuk menyimpan model baru
new_model_path = "D:/Kuliah/skripsi/face-recognition/python-face/model/new-model-cnn-facerecognition.keras"
logging.info(f"New model path: {new_model_path}")

# Threshold confidence untuk deteksi wajah
CONFIDENCE_THRESHOLD = 0.5

# Fungsi untuk mendeteksi wajah
detector = MTCNN()

def detect_face(img):
    logging.info("Detecting face...")
    result = detector.detect_faces(img)
    if result:
        logging.info(f"Face detected with confidence {result[0]['confidence']:.2f}")
        x, y, width, height = result[0]['box']
        cropped_face = img[y:y + height, x:x + width]
        cropped_face_resized = cv2.resize(cropped_face, (50, 50))
        cropped_face_gray = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2GRAY)
        return cropped_face_gray
    logging.info("No face detected.")
    return None

# Fungsi untuk memuat dataset dan memeriksa label baru
def load_and_check_dataset():
    logging.info("Loading dataset...")
    existing_labels = [
        'Angelina Jolie', 
        'Brad Pitt', 
        'Denzel Washington', 
        'Harifin',
        'Hugh Jackman', 
        'Jennifer Lawrence', 
        'Johnny Depp', 
        'Kate Winslet', 
        'Leonardo DiCaprio', 
        'Megan Fox', 
        'Natalie Portman', 
        'Nicole Kidman',
        'Robert Downey Jr',
        'Sandra Bullock',
        'Scarlett Johansson',
        'Tom Cruise',
        'Tom Hanks',
        'Will Smith',
        'Rachmanullah'
    ]
    current_labels = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    new_labels = set(current_labels) - set(existing_labels)

    images, labels = [], []

    for folder in current_labels:
        folder_path = os.path.join(dataset_path, folder)
        logging.info(f"Processing folder: {folder}")
        for file in os.listdir(folder_path):
            if file.endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                face = detect_face(img)
                if face is not None:
                    images.append(face)
                    labels.append(folder)

    logging.info(f"Loaded {len(images)} images with {len(set(labels))} unique labels.")
    return np.array(images), labels, new_labels

# Fungsi untuk membuat model CNN
def cnn_model(input_shape, num_classes):
    logging.info("Creating new CNN model...")
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding="valid", activation="relu"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info("Model created successfully.")
    return model

# API untuk melatih ulang model
@app.route('/retrain', methods=['POST'])
def retrain():
    logging.info("Retrain API called.")
    images, labels, new_labels = load_and_check_dataset()

    if len(new_labels) == 0:
        logging.info("No new labels found. Skipping retraining.")
        return jsonify({'message': 'Tidak ada individu baru. Model tidak perlu dilatih ulang.'})

    # Encoding label
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    # Split dataset
    logging.info("Splitting dataset...")
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels_categorical, test_size=0.15, random_state=42
    )

    # Reshape data
    x_train = x_train.reshape(-1, 50, 50, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 50, 50, 1).astype('float32') / 255.0

    # Buat model baru
    input_shape = x_train[0].shape
    num_classes = len(le.classes_)
    model = cnn_model(input_shape, num_classes)

    # Latih ulang model
    EPOCHS = 10  # Kurangi jumlah epoch untuk mempercepat pelatihan ulang
    BATCH_SIZE = 32

    logging.info(f"Starting model training for {EPOCHS} epochs with batch size {BATCH_SIZE}.")
    history = model.fit(x_train, y_train, validation_split=0.15, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Simpan model baru
    logging.info("Saving new model...")
    model.save(new_model_path)

    logging.info("Model retraining complete.")
    return jsonify({
        'message': 'Model berhasil dilatih ulang.',
        'new_labels': list(new_labels),
        'model_path': new_model_path
    })

if __name__ == '__main__':
    logging.info("Starting Flask app...")
    app.run(debug=True)
