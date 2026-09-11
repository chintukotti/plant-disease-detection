from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import re

app = Flask(__name__)
CORS(app)

# ============ LOAD MODEL & CLASSES ============
model = load_model("model.h5")

with open("classes.json") as f:
    class_indices = json.load(f)

classes = list(class_indices.keys())
print(f"✅ Loaded {len(classes)} classes: {classes}")

# ============ REMEDIES DATABASE ============
REMEDIES = {
    "Pepper__bell___Bacterial_spot": "Apply copper-based bactericides. Remove infected plants. Avoid overhead watering. Use disease-free seeds.",
    "Pepper__bell___healthy": "Your pepper plant is healthy! Continue with proper watering, fertilization, and pest monitoring.",
    "Potato___Early_blight": "Apply fungicides like chlorothalonil or mancozeb. Practice crop rotation. Remove infected leaves promptly.",
    "Potato___healthy": "Your potato plant is healthy! Maintain proper spacing and drainage for best results.",
    "Potato___Late_blight": "Apply fungicides immediately (metalaxyl/mancozeb). Remove and destroy infected plants. Improve air circulation.",
    "Tomato___Target_Spot": "Apply fungicides containing chlorothalonil. Remove infected leaves. Ensure proper plant spacing for airflow.",
    "Tomato___Tomato_mosaic_virus": "Remove and destroy infected plants. Disinfect all tools with bleach. Use resistant tomato varieties.",
    "Tomato___Tomato_YellowLeaf_Curl_Virus": "Control whitefly populations with insecticides. Remove infected plants. Use resistant varieties and reflective mulches.",
    "Tomato_Bacterial_spot": "Apply copper-based sprays early. Remove infected plant debris. Use certified disease-free seeds and transplants.",
    "Tomato_Early_blight": "Apply fungicides (chlorothalonil/mancozeb). Remove lower infected leaves. Mulch around plants to prevent soil splash.",
    "Tomato_healthy": "Your tomato plant is healthy! Keep up good watering practices and monitor for pests regularly.",
    "Tomato_Late_blight": "Apply fungicides immediately. Remove and destroy infected plants. Avoid overhead watering. Improve air circulation.",
    "Tomato_Leaf_Mold": "Improve greenhouse ventilation. Reduce humidity below 85%. Apply fungicides. Remove heavily infected leaves.",
    "Tomato_Septoria_leaf_spot": "Apply fungicides at first sign. Remove infected lower leaves. Avoid overhead watering. Practice crop rotation.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticides or neem oil spray. Increase humidity around plants. Introduce predatory mites as biological control.",
}

# ============ HELPERS ============
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def parse_class_name(name):
    """Parse folder name into plant name and disease name"""
    if name == "Non_leaf":
        return "Non Leaf", "Not a plant"

    if "___" in name:
        parts = name.split("___", 1)
        plant = re.sub(r'_+', ' ', parts[0]).strip()
        disease = re.sub(r'_+', ' ', parts[1]).strip()
    else:
        for p in ["Tomato", "Potato", "Pepper", "Corn", "Apple", "Grape"]:
            if name.startswith(p + "_") or name.startswith(p + " "):
                plant = p
                disease = name[len(p)+1:].replace("_", " ")
                return plant, disease
        plant = name.split("_")[0]
        disease = " ".join(name.split("_")[1:])

    return plant, disease

# ============ ROUTES ============
@app.route("/")
def home():
    return jsonify({
        "message": "Crop Disease API is running 🚀",
        "classes": len(classes),
        "class_names": classes
    })

@app.route("/predict", methods=["POST"])
def predict():
    # Check if file exists
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file).convert("RGB")
    except Exception as e:
        return jsonify({"status": "error", "message": "Invalid image file"}), 400

    # Preprocess and predict
    processed = preprocess(img)
    prediction = model.predict(processed)

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # ---- THRESHOLD CHECK ----
    THRESHOLD = 0.60

    if confidence < THRESHOLD:
        return jsonify({
            "status": "unknown",
            "message": "Could not identify the image clearly. Please upload a clear, close-up photo of a plant leaf.",
            "confidence": round(confidence, 4)
        })

    # ---- NON-LEAF CHECK ----
    if predicted_class == "Non_leaf":
        return jsonify({
            "status": "non_leaf",
            "message": "This image is not a plant leaf. Please upload a photo of a plant leaf for disease detection.",
            "confidence": round(confidence, 4)
        })

    # ---- SUCCESS ----
    plant, disease = parse_class_name(predicted_class)
    is_healthy = "healthy" in disease.lower()
    remedy = REMEDIES.get(predicted_class, "Consult a local agricultural expert for proper diagnosis and treatment.")

    return jsonify({
        "status": "success",
        "plant": plant,
        "disease": disease,
        "is_healthy": is_healthy,
        "confidence": round(confidence, 4),
        "remedy": remedy,
        "raw_class": predicted_class
    })

if __name__ == "__main__":
    app.run(debug=True)