from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
import json

# ---------------------------
# App Initialization
# ---------------------------
app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Paths
# ---------------------------
# Use current file directory for safe paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "global_model.h5")
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ---------------------------
# Load Configuration
# ---------------------------
try:
    with open(CONFIG_PATH, "r") as f:
        model_config = json.load(f)
        metrics = model_config.get("model_metrics", {})
except:
    metrics = {
        "connected_hospitals": 3,
        "training_rounds": 3,
        "hospital_accuracies": {
            "Hospital A": 82,
            "Hospital B": 85,
            "Hospital C": 88
        },
        "current_accuracy": 88,
        "model_type": "CNN - Binary Classification",
        "input_shape": "64x64x3",
        "classes": ["Normal", "Pneumonia"]
    }

# ---------------------------
# Load Model on Startup
# ---------------------------
global_model = None

@app.on_event("startup")
def load_model():
    global global_model
    try:
        if os.path.exists(MODEL_PATH):
            # CPU only
            tf.config.set_visible_devices([], "GPU")
            global_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("[OK] Global model loaded successfully")
        else:
            print("[WARNING] Model file not found:", MODEL_PATH)
            global_model = None
    except Exception as e:
        print("[ERROR] Error loading model:", e)
        global_model = None

# ---------------------------
# Helper Functions
# ---------------------------
def get_training_data():
    training_rounds = metrics.get("training_rounds", 3)
    hospital_accuracies = metrics.get("hospital_accuracies", {})
    hospitals = list(hospital_accuracies.keys())
    training_data = [
        {
            "round": i + 1,
            "hospital": hospitals[i % len(hospitals)],
            "accuracy": accuracy
        }
        for i, (_, accuracy) in enumerate(hospital_accuracies.items())
    ]
    return training_data

def get_hospital_accuracies():
    return metrics.get("hospital_accuracies", {})

# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/training")
def training_endpoint():
    return get_training_data()

@app.get("/status")
def status_endpoint():
    return {
        "connected_hospitals": metrics.get("connected_hospitals", 3),
        "training_rounds": metrics.get("training_rounds", 3),
        "current_accuracy": metrics.get("current_accuracy", 88),
        "model_loaded": global_model is not None,
        "model_type": metrics.get("model_type"),
        "input_shape": metrics.get("input_shape"),
        "classes": metrics.get("classes")
    }

@app.get("/hospital-accuracies")
def hospital_accuracies_endpoint():
    return get_hospital_accuracies()

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if global_model is None:
        return {"error": "Model not loaded"}

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((64, 64))
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = global_model.predict(image_array, verbose=0)
        confidence_score = float(prediction[0][0])
        threshold = 0.46
        predicted_class = "Pneumonia" if confidence_score > threshold else "Normal"

        normal_prob = confidence_score * 100
        pneumonia_prob = (1 - confidence_score) * 100
        max_prob = max(normal_prob, pneumonia_prob)

        return {
            "file_name": file.filename,
            "prediction": predicted_class,
            "confidence": round(max_prob, 2),
            "normal_probability": round(normal_prob, 2),
            "pneumonia_probability": round(pneumonia_prob, 2),
            "raw_prediction": round(confidence_score, 4)
        }

    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Sample Images Endpoints
# ---------------------------
@app.get("/sample-images")
def sample_images_endpoint():
    samples = {"normal": [], "pneumonia": []}
    try:
        normal_path = os.path.join(DATASET_DIR, "normal")
        pneumonia_path = os.path.join(DATASET_DIR, "pneumonia")
        if os.path.exists(normal_path):
            samples["normal"] = os.listdir(normal_path)[:3]
        if os.path.exists(pneumonia_path):
            samples["pneumonia"] = os.listdir(pneumonia_path)[:3]
        return samples
    except Exception as e:
        return {"error": str(e)}

@app.get("/sample-image/{category}/{filename}")
def sample_image_endpoint(category: str, filename: str):
    file_path = os.path.join(DATASET_DIR, category, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    return {"error": "File not found"}

# ---------------------------
# Model Stats
# ---------------------------
@app.get("/model-stats")
def model_stats_endpoint():
    if global_model is None:
        return {"error": "Model not loaded"}
    return {
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "input_shape": [64, 64, 3],
        "output_shape": 1,
        "task": "Binary Classification",
        "classes": ["Normal", "Pneumonia"]
    }

# ---------------------------
# Run Locally
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)