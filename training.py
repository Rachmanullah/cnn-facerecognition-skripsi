import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation
import matplotlib.pyplot as plt
import cv2
from mtcnn import MTCNN
from keras.callbacks import EarlyStopping

# Fungsi untuk mencetak langkah-langkah dengan log
def print_step(message):
    print(f"[STEP] {message}")

# Fungsi untuk menampilkan informasi tentang input data
def display_input_data_info(data, name):
    print(f"[INFO] {name}: shape={data.shape}, dtype={data.dtype}")

# Fungsi untuk mendeteksi dan memotong wajah menggunakan MTCNN
def detect_face(img):
    detector = MTCNN()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img_rgb)

    if result:
        x, y, width, height = result[0]['box']
        x, y = max(0, x), max(0, y)
        cropped_face = img[y:y + height, x:x + width]
        cropped_face = cv2.resize(cropped_face, (50, 50))
        return cropped_face
    else:
        return None

# Fungsi untuk augmentasi gambar
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

    augmented_images = [cv2.warpAffine(img, t, (w, h), borderValue=(255, 255, 255)) for t in transformations]
    intensity_modifications = [cv2.add(img, delta) for delta in [10, 30, -10, -30, 15, 45, -15, -45]]
    return augmented_images + intensity_modifications

# Fungsi untuk menyimpan gambar hasil preprocessing
def save_processed_image(output_dir, folder, filename, img):
    folder_path = os.path.join(output_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, filename)
    img_resized = cv2.resize(img, (50, 50))  # Pastikan ukuran konsisten
    cv2.imwrite(save_path, img_resized)

# Fungsi untuk proses dataset (preprocessing, augmentasi, dan penyimpanan)
def process_dataset(data_dir, output_dir):
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

        # Ambil maksimal 50 gambar atau minimal 10 gambar secara acak
        # if len(files) > 20:
        #     files = np.random.choice(files, 10, replace=False)
        # elif len(files) >= 10:  
        #     files = np.random.choice(files, 10, replace=False)
        # else:
        #     print(f"[WARNING] Folder {folder} dilewati karena kurang dari 10 gambar.")
        #     continue

         # Ambil antara 10 hingga 30 gambar
        if len(files) > 30:
            files = np.random.choice(files, 30, replace=False)
        elif len(files) >= 10:
            files = np.random.choice(files, len(files), replace=False)

        for filename in files:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                # Resize gambar ke 50x50 untuk konsistensi
                img_resized = cv2.resize(img, (50, 50))
                face = detect_face(img_resized)  # Deteksi wajah setelah resize
                if face is not None:
                    # save_processed_image(output_dir, folder, filename, face)
                    images.append(face)
                    names.append(folder)

                    # Augmentasi
                    augmented_images = img_augmentation(face)
                    for j, aug_img in enumerate(augmented_images):
                        if aug_img.shape == (50, 50, 3):  # Validasi dimensi hasil augmentasi
                            # save_processed_image(output_dir, folder, f"aug_{j}_{filename}", aug_img)
                            images.append(aug_img)
                            names.append(folder)
                        else:
                            print(f"[WARNING] Augmentasi menghasilkan dimensi {aug_img.shape}. Gambar dilewati.")

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

# Fungsi untuk membuat CNN
def cnn_model(input_shape, num_classes):
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

# Main Execution
data_dir = "../front-facerecognition/public/imagesFace/daftar"
output_dir = "dataset_pre"
model_dir = "model_cnn"

# Contoh penggunaan, misalnya, untuk membaca nama folder sebagai label
if os.path.exists(data_dir):
    labels = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
    print("Labels ditemukan:", labels)
else:
    print(f"Direktori {data_dir} tidak ditemukan!")
    
# Proses dataset
images, names = process_dataset(data_dir, output_dir)

# Validasi konsistensi dimensi dataset
valid_images = []
valid_names = []

for i, img in enumerate(images):
    if img.shape == (50, 50, 3):  # Pastikan dimensi sesuai
        valid_images.append(img)
        valid_names.append(names[i])
    else:
        print(f"[WARNING] Gambar ke-{i} memiliki dimensi {img.shape} dan dilewati.")

images = valid_images
names = valid_names

# Balance data
names, images = balance_data(names, images, n=200)

# Encoding label
print_step("Memulai encoding label dan konversi ke one-hot")
le = LabelEncoder()
le.fit(names)
labels = le.classes_
name_vec = le.transform(names)
categorical_name_vec = to_categorical(name_vec)

# Split dataset
print_step("Memulai split dataset")
x_train, x_test, y_train, y_test = train_test_split(
    np.array(images, dtype=np.float32),
    np.array(categorical_name_vec),
    test_size=0.15,
    random_state=42
)

# Debug sebelum reshape
try:
    print(f"[DEBUG] Dimensi x_train sebelum reshape: {x_train.shape}")
    x_train = x_train.reshape(x_train.shape[0], 50, 50, 3)
    x_test = x_test.reshape(x_test.shape[0], 50, 50, 3)
    print(f"[INFO] Reshape berhasil: x_train {x_train.shape}, x_test {x_test.shape}")
except ValueError as e:
    print(f"[ERROR] Reshape gagal: {e}")
    print(f"[INFO] Periksa total elemen x_train: {x_train.size}")

# Early Stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',    # Monitor 'val_loss' untuk mencegah overfitting
    patience=5,            # Berhenti jika tidak ada perbaikan selama 5 epoch
    restore_best_weights=True  # Kembalikan bobot terbaik
)

# Train CNN
print_step("Memulai pelatihan model CNN")
input_shape = x_train[0].shape
num_classes = len(labels)
model = cnn_model(input_shape, num_classes)
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    shuffle=True,
    validation_split=0.15,
    # callbacks=[early_stopping]
)

# Save Model
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model_cnn_face_recognition-test3.keras")
model.save(model_path)
print_step(f"Model disimpan ke {model_path}")

# Evaluate Model
print_step("Evaluasi model")
def evaluate_model(history):
    metrics = [['accuracy', 'val_accuracy'], ['loss', 'val_loss']]
    for metric in metrics:
        plt.plot(history.history[metric[0]])
        plt.plot(history.history[metric[1]])
        plt.title(f'Model {metric[0]}')
        plt.xlabel('Epoch')
        plt.ylabel(metric[0])
        plt.legend(['Training', 'Validation'], loc='best') 
        plt.show()

evaluate_model(history)
