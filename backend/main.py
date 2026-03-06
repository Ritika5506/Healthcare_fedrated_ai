from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample training data
training_data = [
    {"round": 1, "hospital": "Hospital A", "accuracy": 78},
    {"round": 2, "hospital": "Hospital B", "accuracy": 82},
    {"round": 3, "hospital": "Hospital C", "accuracy": 86},
]

@app.get("/training")
def get_training_data():
    return training_data

@app.get("/status")
def system_status():
    return {
        "connected_hospitals": 3,
        "training_rounds": 3,
        "current_accuracy": 86
    }