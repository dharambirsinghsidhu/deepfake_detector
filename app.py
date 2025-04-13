import os
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image

# ✅ Fix Keras activation issue
tf.keras.utils.get_custom_objects().clear()

# ✅ Model path and Google Drive ID
MODEL_PATH = "deepfake_model.h5"
GOOGLE_DRIVE_FILE_ID = "1xlTb2ToE82F4wAzTAJxRIejW4lX8bMEy"  # Replace with actual file ID

# ✅ Download the model if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        import gdown
        # Construct the Google Drive URL for direct download
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ✅ Ensure model is available
download_model()

# ✅ Load the trained model
model = load_model(MODEL_PATH)

# ✅ Function to compute entropy
def calculate_entropy(image):
    """Compute the entropy of a grayscale image."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Convert from float to uint8

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))  # Avoid log(0) issue

    return entropy

# ✅ Function to preprocess uploaded images
def preprocess_image(image):
    """Preprocess the uploaded image for the model."""
    image = image.resize((256, 256))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    entropy_value = calculate_entropy(image)  # Compute entropy
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    entropy_value = np.array([[entropy_value]])  # Reshape to (1, 1)
    return image, entropy_value

# ✅ Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# ✅ Route: Home Page (Upload Image)
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ✅ Route: Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    try:
        # ✅ Save the uploaded file
        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        # ✅ Read and preprocess the image
        image = Image.open(filepath).convert("RGB")
        processed_image, entropy_value = preprocess_image(image)

        # ✅ Model Prediction (using both image & entropy)
        prediction = model.predict([processed_image, entropy_value])[0][0]
        label = "Real" if prediction > 0.5 else "Fake"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # ✅ Redirect to result page with the prediction results
        return render_template(
            "result.html",
            label=label,
            confidence=f"{confidence:.2%}",
            image_path=filepath,  # Pass image path to display it
        )
    except Exception as e:
        return f"Processing error: {str(e)}"

# ✅ Run Flask App Locally
if __name__ == "__main__":
    app.run(debug=False)  # Runs on http://127.0.0.1:5000
