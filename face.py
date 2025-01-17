from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import cv2
from mtcnn import MTCNN
from flask_cors import CORS
import logging

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.info("API is starting...")

# Load the Keras model
model = load_model('D:/Kuliah/skripsi/face-recognition/python-face/model/model-cnn-facerecognition-9.keras')
app.logger.info("Model loaded successfully.")

# Initialize MTCNN for face detection
detector = MTCNN()

# Labels for prediction
labels = [
    '2118001', '2118002', '2118003', '2118131',
    '2118004', '2118005', '2118006', '2118007',
    '2118008', '2118009', '2118010', '2118011',
    '2118012', '2118013', '2118014', '2118015',
    '2118016', '2118017', '2118100'
]

def detect_face(frame):
    result = detector.detect_faces(frame)
    if result:
        x, y, width, height = result[0]['box']
        cropped_face = frame[y:y + height, x:x + width]
        cropped_face_resized = cv2.resize(cropped_face, (50, 50))
        cropped_face_gray = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2GRAY)
        return cropped_face_gray, (x, y, width, height)  # Return face and bounding box
    return None, None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Process input image
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        app.logger.info("Image received and decoded successfully.")

        # Detect face
        face, box = detect_face(img)
        if face is None:
            app.logger.warning("No face detected in the image.")
            return jsonify({'prediction': 'Face not detected'}), 400

        # Prepare face for model prediction
        face = face.reshape(1, 50, 50, 1)  # 1 channel for grayscale
        prediction = model.predict(face)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # Determine prediction label
        if confidence < 0.5:
            predicted_name = "tidak dikenali"
            app.logger.warning("Low confidence prediction.")
        else:
            predicted_name = labels[class_idx]
            app.logger.info(f"Prediction successful: {predicted_name} with confidence {confidence:.2f}.")

        # Return response
        return jsonify({
            'prediction': predicted_name,
            'confidence': confidence,
            'box': box
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.logger.info("Starting the API server...")
    app.run(debug=True)
