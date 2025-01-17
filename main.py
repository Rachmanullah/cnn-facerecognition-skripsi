from flask import Flask, request, jsonify
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.callbacks import EarlyStopping
from mtcnn import MTCNN
from flask_cors import CORS
import cv2
import json
import traceback
import time

# Flask app initialization
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://907d-103-154-144-186.ngrok-free.app"], "supports_credentials":False}})

# Constants
DATA_DIR = "../front-facerecognition/public/imagesFace/daftar"
MODEL_DIR = "model"
MODEL_NAME = "model_new_rgb_api.keras"
JSON_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),  # Lokasi file Python ini
    "../front-facerecognition/public/nim_labels.json"
))
MODEL_PREDICT = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model_cnn/model_cnn_face_recognition-testAPIWeb.keras')

# Utility functions
def log_step(message):
    print(f"[LOG] {message}")

def detect_face(img):
    log_step("Detecting face...")
    detector = MTCNN()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img_rgb)

    if result:
        x, y, width, height = result[0]['box']
        x, y = max(0, x), max(0, y)
        cropped_face = img[y:y + height, x:x + width]
        cropped_face = cv2.resize(cropped_face, (50, 50))
        return cropped_face
    return None

def detect_face_predict(frame):
    detector = MTCNN()
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

def img_augmentation(img):
    log_step("Performing image augmentation...")
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    transformations = [
        cv2.getRotationMatrix2D(center, angle, 1.0) for angle in [5, -5, 10, -10]
    ] + [
        np.float32([[1, 0, tx], [0, 1, 0]]) for tx in [3, -3, 6, -6]
    ] + [
        np.float32([[1, 0, 0], [0, 1, ty]]) for ty in [3, -3, 6, -6]
    ]

    augmented_images = [cv2.warpAffine(img, t, (w, h), borderValue=(255, 255, 255)) for t in transformations]
    intensity_modifications = [cv2.add(img, delta) for delta in [10, 30, -10, -30, 15, 45, -15, -45]]
    return augmented_images + intensity_modifications


def balance_data(names, images, n=200):
    log_step("Balancing data...")
    balanced_names, balanced_images = [], []
    unique_labels = np.unique(names)

    for label in unique_labels:
        label_indices = np.where(np.array(names) == label)[0]
        selected_indices = np.random.choice(label_indices, min(n, len(label_indices)), replace=False)
        balanced_names.extend([names[i] for i in selected_indices])
        balanced_images.extend([images[i] for i in selected_indices])

    return balanced_names, balanced_images

def cnn_model(input_shape, num_classes):
    log_step("Creating CNN model...")
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding="valid", activation="relu"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Labels Loader
def load_labels(JSON_PATH):
    if not os.path.exists(JSON_PATH):
        log_step(f"[ERROR] File JSON tidak ditemukan: {JSON_PATH}")
        return []
    try:
        with open(JSON_PATH, "r") as file:
            labels = json.load(file)
            log_step(f"[INFO] Label berhasil dimuat: {labels}")
            return labels
    except json.JSONDecodeError as e:
        log_step(f"[ERROR] Gagal memuat file JSON: {e}")
        return []

labels = load_labels(JSON_PATH)

# ROUTE TRAINING
@app.route('/training', methods=['POST'])
def train_model():
    try:
        log_step("Starting training process...")

        # Load dataset
        if not os.path.exists(DATA_DIR):
            log_step("Data directory not found!")
            return jsonify({"error": "Data directory not found"}), 400

        images, names = [], []
        for folder in os.listdir(DATA_DIR):
            folder_path = os.path.join(DATA_DIR, folder)
            if not os.path.isdir(folder_path):
                continue

            files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            log_step(f"Folder '{folder}' contains {len(files)} images.")

            files = np.random.choice(files, min(20, len(files)), replace=False)  # Random selection if more than 30 images
            log_step(f"Processing {len(files)} images from folder '{folder}'...")

            for filename in files:
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)

                if img is not None:
                    img_resized = cv2.resize(img, (50, 50))
                    face = detect_face(img_resized)
                    if face is not None:
                        images.append(face)
                        names.append(folder)

                        # Augment the image
                        augmented_images = img_augmentation(face)
                        images.extend(augmented_images)
                        names.extend([folder] * len(augmented_images))

            log_step(f"After augmentation, folder '{folder}' has {len(files) * 21} images (1 original + 20 augmentations per image).")

        total_images_before_balance = len(images)
        log_step(f"Total images after augmentation across all folders: {total_images_before_balance}.")

        # Balance data
        names, images = balance_data(names, images, n=200)
        log_step(f"Total images after balancing: {len(images)}.")

        # Encode labels
        log_step("Encoding labels...")
        le = LabelEncoder()
        le.fit(names)
        labels = le.classes_
        name_vec = le.transform(names)
        categorical_name_vec = to_categorical(name_vec)

        # Split dataset
        log_step("Splitting dataset into training and testing...")
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images, dtype=np.float32),
            np.array(categorical_name_vec),
            test_size=0.15,
            random_state=42
        )
        log_step(f"Training dataset size: {len(x_train)}. Testing dataset size: {len(x_test)}.")

        # Reshape data
        x_train = x_train.reshape(x_train.shape[0], 50, 50, 3)
        x_test = x_test.reshape(x_test.shape[0], 50, 50, 3)

        # Train CNN
        log_step("Training CNN model...")
        input_shape = x_train[0].shape
        num_classes = len(labels)
        model = cnn_model(input_shape, num_classes)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        start_time = time.time()
        history = model.fit(
            x_train, y_train,
            epochs=20,
            batch_size=32,
            shuffle=True,
            validation_split=0.15,
            callbacks=[early_stopping]
        )
         # Save model
        log_step("Saving trained model...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        model.save(model_path)

        # Step 7: Evaluate Model
        log_step("Evaluasi model")
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        report = classification_report(y_test_classes, y_pred_classes, target_names=labels, output_dict=True)
        confusion = confusion_matrix(y_test_classes, y_pred_classes).tolist()

        # Hitung akurasi pengujian
        test_accuracy = np.mean(y_test_classes == y_pred_classes)

        end_time = time.time() 
        training_duration = end_time - start_time  # Dalam detik

        log_step("Training process completed successfully!")
        return jsonify({
            "message": "Model training completed successfully",
            "training_duration_seconds": training_duration,
            "training_duration_readable": f"{int(training_duration // 60)} minutes {int(training_duration % 60)} seconds",
            "model_path": model_path,
            "num_epochs": len(history.history['accuracy']),
            "training_history": {
                "accuracy": history.history['accuracy'],
                "val_accuracy": history.history['val_accuracy'],
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss']
            },
            "test_accuracy": test_accuracy,
            "classification_report": report,
            "confusion_matrix": confusion
        })
    except Exception as e:
        log_step(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    
# ROUTE PREDICTIONS
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image'].read()  # ambil gambar dari request
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # Decode image in RGB format

    face, box = detect_face_predict(img)
    if face is None:
        return jsonify({'prediction': 'Face not detected'})

    # Prepare wajah untuk model prediction (already grayscale)
    face = face.reshape(1, 50, 50, 3)  # 1 channel for grayscale

    prediction = MODEL_PREDICT.predict(face)
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