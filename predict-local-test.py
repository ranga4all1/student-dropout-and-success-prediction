#!/usr/bin/env python
# coding: utf-8

# Load and use model for prediction

# import required libraries
print(f"Importing required libraries...")
import pickle
import pandas as pd


# Load model
print(f"Loading model...")
model_file = 'model_rf.bin'

with open(model_file, 'rb') as f_in:
    scaler, model_rf = pickle.load(f_in)


# Test on new student example
student_data = {
    'curricular_units_2nd_sem_(approved)': [8],
    'curricular_units_2nd_sem_(grade)': [14.07125],
    'curricular_units_1st_sem_(approved)': [8],
    'curricular_units_1st_sem_(grade)': [14.07125],
    'tuition_fees_up_to_date': [1],
    'scholarship_holder': [1],
    'age_at_enrollment': [19],
    'debtor': [0],
    'gender': [0],
    'application_mode': [1],
    'curricular_units_2nd_sem_(enrolled)': [8],
    'curricular_units_1st_sem_(enrolled)': [8],
    'displaced': [1]
}


# Create DataFrame
student = pd.DataFrame(student_data)
X = scaler.transform(student)

# Get predictions
y_pred = model_rf.predict(X)
y_pred_proba = model_rf.predict_proba(X)

print("input:", student_data)
print("-" * 50)

# Define the mapping
mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
# Map predictions to labels
y_pred_labels = [mapping[pred] for pred in y_pred]
for i, (label, proba) in enumerate(zip(y_pred_labels, y_pred_proba)):
    print(f"Student {i+1}: Prediction = {label}, Probabilities = {proba}")
    print("-" * 50)
