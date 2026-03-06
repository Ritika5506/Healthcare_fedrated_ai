import flwr as fl
import pandas as pd
from model.model import create_model

# Load dataset
data = pd.read_csv("../dataset/diabetes.csv")

# Split features and target
X = data.drop("Outcome", axis=1).values
y = data["Outcome"].values

# Create model
model = create_model()


class HospitalClient(fl.client.NumPyClient):

    # Send model parameters to server
    def get_parameters(self, config=None):
        return model.get_weights()

    # Train model locally
    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X, y, epochs=1, batch_size=32, verbose=1)
        return model.get_weights(), len(X), {}

    # Evaluate model
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X, y, verbose=0)
        return loss, len(X), {"accuracy": accuracy}


# Start hospital client
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=HospitalClient(),
)
