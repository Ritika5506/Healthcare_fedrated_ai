from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image
import io
import json

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
global_model = None
global_model_path = os.path.join(project_root, "global_model.h5")

# Load configuration
config_path = os.path.join(project_root, "model_config.json")

try:
    with open(config_path, "r") as f:
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
# Load Model on Server Start
# ---------------------------
@app.on_event("startup")
def load_model():
    global global_model
    try:
        if os.path.exists(global_model_path):
            global_model = tf.keras.models.load_model(
                global_model_path,
                compile=False
            )
            print("[OK] Global model loaded successfully")
        else:
            print("[WARNING] Model file not found:", global_model_path)
            global_model = None
    except Exception as e:
        print("[ERROR] Error loading model:", e)
        global_model = None


# ---------------------------
# Training Data
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
def get_training_data_endpoint():
    return get_training_data()


@app.get("/status")
def system_status():
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
def get_hospital_accuracies_endpoint():
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
# Sample Images
# ---------------------------

@app.get("/sample-images")
def get_sample_images():

    sample_images = {
        "normal": [],
        "pneumonia": []
    }

    try:
        normal_path = os.path.join(project_root, "dataset/normal")
        pneumonia_path = os.path.join(project_root, "dataset/pneumonia")

        if os.path.exists(normal_path):
            sample_images["normal"] = os.listdir(normal_path)[:3]

        if os.path.exists(pneumonia_path):
            sample_images["pneumonia"] = os.listdir(pneumonia_path)[:3]

        return sample_images

    except Exception as e:
        return {"error": str(e)}


@app.get("/sample-image/{category}/{filename}")
def get_sample_image(category: str, filename: str):

    file_path = os.path.join(project_root, f"dataset/{category}", filename)

    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")

    return {"error": "File not found"}


# ---------------------------
# Model Stats
# ---------------------------

@app.get("/model-stats")
def get_model_stats():

    if global_model is None:
        return {"error": "Model not loaded"}

    return {
        "model_loaded": True,
        "model_path": global_model_path,
        "input_shape": [64, 64, 3],
        "output_shape": 1,
        "task": "Binary Classification",
        "classes": ["Normal", "Pneumonia"]
    }


# ---------------------------
# Local Run
# ---------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)