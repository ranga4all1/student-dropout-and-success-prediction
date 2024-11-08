#!/usr/bin/env python
# coding: utf-8

# Load and use model for prediction

# import required libraries
print(f"Importing required libraries...")
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


# Load model
print(f"Loading model...")
model_file = 'model_rf.bin'

with open(model_file, 'rb') as f_in:
    scaler, model_rf = pickle.load(f_in)


app = Flask("dropout")


@app.route("/predict", methods=["POST"])
def predict():
    # Get and validate request data
    student_data = request.get_json()
    # Create DataFrame
    student = pd.DataFrame(student_data)

    # ransform features
    X = scaler.transform(student)

    # Get predictions
    y_pred = model_rf.predict(X)
    y_pred_proba = model_rf.predict_proba(X)

    # Define the mapping
    mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
    # Create results with proper formatting
    result = {
        'predictions by model': [
            {
                'student_id': i + 1,
                'predicted_status': mapping[pred],
                'probabilities': {
                    'Dropout': float(probs[0]),
                    'Enrolled': float(probs[1]),
                    'Graduate': float(probs[2])
                }
            }
            for i, (pred, probs) in enumerate(zip(y_pred, y_pred_proba))
        ]
    }
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
