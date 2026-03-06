import flwr as fl
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import create_model
from blockchain import hash_parameters

# Load hospital A dataset
train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    "hospitals/hospital_C",
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# CNN Model
model = create_model()

# Flower Client
class HospitalClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train_data, epochs=1)
        updated_parameters = model.get_weights()
        hash_val = hash_parameters(updated_parameters)
        return updated_parameters, train_data.samples, {"hash": hash_val, "hospital_id": "C"}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(train_data)
        return loss, train_data.samples, {"accuracy": accuracy}

# read server address written by server
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server_address.txt"), "r") as f:
    server_address = f.read().strip()

# Start client
fl.client.start_numpy_client(
    server_address=server_address,
    client=HospitalClient(),
)