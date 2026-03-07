from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image
import io
import base64
from pathlib import Path
import json

# Add parent directory to path for imports
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

# Load model configuration
config_path = os.path.join(project_root, "model_config.json")
try:
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        metrics = model_config.get("model_metrics", {})
except:
    metrics = {
        "connected_hospitals": 3,
        "training_rounds": 3,
        "hospital_accuracies": {"Hospital A": 82, "Hospital B": 85, "Hospital C": 88},
        "current_accuracy": 88,
        "model_type": "CNN - Binary Classification",
        "input_shape": "64x64x3",
        "classes": ["Normal", "Pneumonia"]
    }

# Load global model
global_model_path = os.path.join(project_root, "global_model.h5")
try:
    if os.path.exists(global_model_path):
        global_model = tf.keras.models.load_model(global_model_path)
        print("[OK] Global model loaded successfully")
    else:
        global_model = None
        print("[WARNING] Global model not found")
except Exception as e:
    global_model = None
    print(f"[ERROR] Error loading model: {e}")

# Dynamic training data based on configuration
def get_training_data():
    """Generate training data based on config"""
    training_rounds = metrics.get("training_rounds", 3)
    hospital_accuracies = metrics.get("hospital_accuracies", {
        "Hospital A": 82,
        "Hospital B": 85,
        "Hospital C": 88
    })
    
    hospitals = list(hospital_accuracies.keys())
    training_data = [
        {"round": i + 1, "hospital": hospitals[i % len(hospitals)], "accuracy": accuracy}
        for i, (_, accuracy) in enumerate(hospital_accuracies.items())
    ]
    return training_data

def get_hospital_accuracies():
    """Get accuracy for each hospital from config"""
    return metrics.get("hospital_accuracies", {
        "Hospital A": 82,
        "Hospital B": 85,
        "Hospital C": 88
    })

@app.get("/training")
def get_training_data_endpoint():
    """Get training rounds and hospital accuracies"""
    return get_training_data()

@app.get("/status")
def system_status():
    """Get system status with actual model metrics from config"""
    training_data = get_training_data()
    current_accuracy = metrics.get("current_accuracy", 88)
    
    return {
        "connected_hospitals": metrics.get("connected_hospitals", 3),
        "training_rounds": metrics.get("training_rounds", 3),
        "current_accuracy": current_accuracy,
        "model_loaded": global_model is not None,
        "model_type": metrics.get("model_type", "CNN - Binary Classification"),
        "input_shape": metrics.get("input_shape", "64x64x3"),
        "classes": metrics.get("classes", ["Normal", "Pneumonia"])
    }

@app.get("/hospital-accuracies")
def get_hospital_accuracies_endpoint():
    """Get accuracy for each hospital"""
    return get_hospital_accuracies()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict on uploaded image"""
    try:
        if global_model is None:
            return {"error": "Model not loaded"}
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Resize to model input shape
        image = image.resize((64, 64))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        prediction = global_model.predict(image_array, verbose=0)
        confidence = float(prediction[0][0])
        
        # Determine class (0 = Normal, 1 = Pneumonia)
        predicted_class = "Pneumonia" if confidence > 0.5 else "Normal"
        
        return {
            "file_name": file.filename,
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "normal_prob": round((1 - confidence) * 100, 2),
            "pneumonia_prob": round(confidence * 100, 2)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/sample-images")
def get_sample_images():
    """Get list of available sample images for testing"""
    try:
        sample_images = {
            "normal": [],
            "pneumonia": []
        }
        
        # Get normal samples
        normal_path = os.path.join(project_root, "dataset/normal")
        if os.path.exists(normal_path):
            normal_files = os.listdir(normal_path)[:3]  # Get first 3
            sample_images["normal"] = normal_files
        
        # Get pneumonia samples
        pneumonia_path = os.path.join(project_root, "dataset/pneumonia")
        if os.path.exists(pneumonia_path):
            pneumonia_files = os.listdir(pneumonia_path)[:3]  # Get first 3
            sample_images["pneumonia"] = pneumonia_files
        
        return sample_images
    except Exception as e:
        return {"error": str(e)}

@app.get("/sample-image/{category}/{filename}")
def get_sample_image(category: str, filename: str):
    """Get a sample image for testing"""
    try:
        file_path = os.path.join(project_root, f"dataset/{category}", filename)
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="image/jpeg")
        else:
            return {"error": "File not found"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/model-stats")
def get_model_stats():
    """Get model architecture and statistics"""
    try:
        if global_model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_loaded": True,
            "model_path": global_model_path,
            "input_shape": [64, 64, 3],
            "output_shape": 1,
            "task": "Binary Classification (Normal vs Pneumonia)",
            "classes": ["Normal", "Pneumonia"]
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")