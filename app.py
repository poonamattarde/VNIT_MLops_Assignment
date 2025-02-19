from flask import Flask, request, jsonify
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from pyngrok import ngrok

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the ML-Flow experiment
mlflow.set_experiment("iris_classification")

app = Flask(__name__)

# Define the best model parameters endpoint
@app.route("/best_model_parameters", methods=["GET"])
def get_best_model_parameters():
    with mlflow.start_run():
        # Get the best hyperparameters from the ML-Flow experiment
        best_run = mlflow.search_runs(order_by=["metrics.accuracy DESC"])[0]
        best_hyperparameters = best_run.data.params
        return jsonify(best_hyperparameters)

# Define the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    with mlflow.start_run():
        # Get the input features from the request payload
        input_features = request.get_json()["features"]
        # Load the best model from the ML-Flow experiment
        best_run = mlflow.search_runs(order_by=["metrics.accuracy DESC"])[0]
        best_model = mlflow.sklearn.load_model(best_run.info.artifact_uri + "/model")
        # Make predictions using the best model
        predictions = best_model.predict(input_features)
        return jsonify({"predictions": predictions.tolist()})

# Define the training endpoint
@app.route("/train", methods=["POST"])
def train():
    with mlflow.start_run():
        # Get the hyperparameters from the request payload
        hyperparameters = request.get_json()["hyperparameters"]
        # Train a new model using the provided hyperparameters
        with mlflow.start_run():
            mlflow.log_param("n_estimators", hyperparameters["n_estimators"])
            mlflow.log_param("max_depth", hyperparameters["max_depth"])
            model = RandomForestClassifier(n_estimators=hyperparameters["n_estimators"], max_depth=hyperparameters["max_depth"])
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
        return jsonify({"message": "Training complete"})

# Expose the Flask app to the internet using ngrok
!pip install pyngrok
http_tunnel = ngrok.connect(5003)
public_url = http_tunnel.public_url
print("Public URL:", public_url)

app.run(port=5003)
