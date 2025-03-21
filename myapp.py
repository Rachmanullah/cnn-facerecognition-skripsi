import os
import threading
import sys
from flask import Flask, jsonify, request, send_from_directory
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
from tensorflow.keras.callbacks import Callback

app = Flask(__name__)

# Kode ini mengatur kebijakan CORS (Cross-Origin Resource Sharing) untuk aplikasi Flask. Hal ini memungkinkan aplikasi untuk menerima permintaan dari domain yang berbeda.
# - `origins`: Daftar URL yang diizinkan untuk mengakses API.
# - `supports_credentials`: Memungkinkan pengiriman kredensial (seperti cookie).
# - `expose_headers` dan `allow_headers`: Menentukan header khusus yang boleh digunakan atau diekspos dalam komunikasi.
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://face-recognition-lyart.vercel.app",  # URL produksi Next.js
            "http://localhost:3000"  # URL pengembangan lokal
        ],
        "supports_credentials": True
    }
}, expose_headers=["Content-Type", "ngrok-skip-browser-warning"], allow_headers=["Content-Type", "ngrok-skip-browser-warning"])

# Constants
DATA_DIR = "public/imagesFace/daftar"
MODEL_DIR = "model"
MODEL_NAME = "model_new_gray_api_test.keras"
JSON_PATH = os.path.abspath("public/nim_labels.json")
MODEL_PREDICT = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model/model_new_gray_api_test.keras')
TRAINING_RESULTS_FILE = 'training_results.json'

# Variabel global untuk status pelatihan
training_status = {
    "status": "idle",  # idle, running, completed, error
    "progress": 0,      # Persentase progres (0-100)
    "message": "",
    "error": None
}

# function untuk memperbarui status dari training model
def update_status(status, progress, message): 
    """Update the global training status."""
    training_status["status"] = status
    training_status["progress"] = progress
    training_status["message"] = message

# Custom callback untuk update status per epoch
class TrainingStatusUpdater(Callback):
    def __init__(self, total_epochs, start_progress=50, end_progress=90):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_progress = start_progress
        self.end_progress = end_progress

    def on_epoch_end(self, epoch, logs=None):
        epoch_progress = self.start_progress + int(((epoch + 1) / self.total_epochs) * (self.end_progress - self.start_progress))
        accuracy = logs.get("accuracy", 0.0)
        loss = logs.get("loss", 0.0)
        update_status("running", epoch_progress, f"Epoch {epoch + 1}/{self.total_epochs} selesai. Akurasi: {accuracy:.2f}, Loss: {loss:.2f}")

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

def process_training():
    """Proses pelatihan dijalankan di background."""
    try:
        update_status("running", 0, "Proses pelatihan dimulai")
        start_time = time.time()

        # Step 1: Load dan Preprocess Dataset
        update_status("running", 5, "Memuat dan memproses dataset")
        if not os.path.exists(DATA_DIR):
            update_status("error", 0, "Directory data tidak ditemukan")
            return

        # Step 2: Load dan Preprocess Dataset
        update_status("running", 10, "Memuat dan memproses dataset")
        images, names = process_dataset(DATA_DIR)

        # Validasi dimensi gambar
        valid_images, valid_names = [], []
        for i, img in enumerate(images):
            if img.shape == (50, 50):
                valid_images.append(img[:, :, np.newaxis])
                valid_names.append(names[i])
        images, names = valid_images, valid_names

        if not images or not names:
            update_status("error", 10, "Tidak ada data yang valid setelah preprocessing")
            return

        # Step 3: Balance Dataset
        update_status("running", 20, "Menyeimbangkan dataset")
        names, images = balance_data(names, images, n=200)

        # Step 4: Encode Labels
        update_status("running", 30, "Encoding label dan konversi ke one-hot")
        le = LabelEncoder()
        le.fit(names)
        labels = le.classes_
        name_vec = le.transform(names)
        categorical_name_vec = to_categorical(name_vec)

        # Step 5: Split Dataset
        update_status("running", 40, "Membagi dataset menjadi training dan testing")
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images, dtype=np.float32),
            np.array(categorical_name_vec),
            test_size=0.15,
            random_state=42
        )

        # Step 6: Train Model
        update_status("running", 50, "Pelatihan model CNN dimulai")
        input_shape = x_train[0].shape
        num_classes = len(labels)
        model = cnn_model(input_shape, num_classes)

        total_epochs = 15

        history = model.fit(
            x_train, y_train,
            epochs=total_epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.15,
            callbacks=[TrainingStatusUpdater(total_epochs)]
        )

        # Step 7: Save Model
        update_status("running", 90, "Menyimpan model")
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        model.save(model_path)

        # Step 8: Evaluate Model
        update_status("running", 95, "Evaluasi model")
        unique_labels, counts = np.unique(y_test.argmax(axis=1), return_counts=True)
        subset_x, subset_y, subset_true_labels = [], [], []

        for label in unique_labels:
            indices = np.where(y_test.argmax(axis=1) == label)[0]
            selected_indices = np.random.choice(indices, min(20, len(indices)), replace=False)
            subset_x.extend(x_test[selected_indices])
            subset_y.extend(y_test[selected_indices])
            subset_true_labels.extend([label] * len(selected_indices))

        subset_x = np.array(subset_x)
        subset_y = np.array(subset_y)
        subset_true_labels = np.array(subset_true_labels)

        subset_predictions = model.predict(subset_x)
        subset_predicted_labels = np.argmax(subset_predictions, axis=1)

        correct = (subset_predicted_labels == subset_true_labels).sum()
        report = classification_report(
            subset_true_labels,
            subset_predicted_labels,
            target_names=[labels[i] for i in unique_labels],
            output_dict=True
        )
        confusion = confusion_matrix(subset_true_labels, subset_predicted_labels).tolist()

        end_time = time.time()
        training_duration = end_time - start_time

        training_results = {
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
            "test_accuracy": correct / len(subset_true_labels),
            "classification_report": report,
            "confusion_matrix": confusion
        }

        with open(TRAINING_RESULTS_FILE, 'w') as f:
            json.dump(training_results, f)

        # Final Step: Selesai
        update_status("completed", 100, "Pelatihan selesai")

    except Exception as e:
        update_status("error", 0, str(e))

@app.route('/public/<path:filename>', methods=['GET'])
def serve_public_file(filename):
    return send_from_directory('public', filename)

@app.route('/', methods=['GET'])
def home():
    return 'hallo sobat'

@app.route('/register', methods=['POST'])
def register():
    try:
        nim = request.form.get("nim")
        nama = request.form.get("nama")
        photos = request.files.getlist("photos")

        if not nim or not nama or not photos:
            return jsonify({"status": "error", "message": "Invalid data. Ensure nim, nama, and photos are provided."}), 400

        user_dir = os.path.join(DATA_DIR, nim)
        photo_details = []

        for photo in photos:
            timestamp = int(time.time() * 1000)
            filename = f"{timestamp}_{photo.filename}"
            file_path = save_image(photo, user_dir, filename)

            photo_details.append({
                "filename": filename,
                "path": f"/public/imagesFace/daftar/{nim}/{filename}",
            })

        # Update JSON labels
        nim_labels = []
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, "r") as json_file:
                nim_labels = json.load(json_file)

        if nim not in nim_labels:
            nim_labels.append(nim)
            nim_labels.sort()

        with open(JSON_PATH, "w") as json_file:
            json.dump(nim_labels, json_file, indent=4)

        return jsonify({
            "status": "success",
            "message": "Photos and labels saved successfully.",
            "data": {
                "nim": nim,
                "nama": nama,
                "photos": photo_details
            }
        })

    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/training', methods=['POST'])
def train_model():
    if training_status["status"] == "running":
        return jsonify({"status": "error", "message": "Pelatihan sudah berjalan"}), 400

    threading.Thread(target=process_training).start()
    return jsonify({"status": "running", "message": "Pelatihan dimulai"}), 200

# cek status training
@app.route('/training_status', methods=['GET'])
def get_training_status():
    print("Received request for /training_status")
    response = jsonify(training_status)
    response.headers['Content-Type'] = 'application/json'
    return response

# Endpoint untuk mendapatkan hasil pelatihan
@app.route('/training_result', methods=['GET'])
def training_result():
    if not os.path.exists(TRAINING_RESULTS_FILE):
        return jsonify({"status": "error", "message": "No training result available."}), 404

    with open(TRAINING_RESULTS_FILE, 'r') as f:
        training_results = json.load(f)
    update_status("idle", 0, "")
    response = jsonify(training_results)
    response.headers['Content-Type'] = 'application/json'
    return response

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
    face = face.reshape(1, 50, 50, 1)  # 1 channel for grayscale

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

if __name__ == "__main__":
    app.run(debug=True)

