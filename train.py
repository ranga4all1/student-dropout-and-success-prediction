#!/usr/bin/env python
# coding: utf-8

# Train and save model

# import required libraries
print(f"Importing required libraries...")
import numpy as np
import pandas as pd

import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer


# load data
print(f"Loading data...")
df = pd.read_csv("data/dataset.csv")

print(f"Pre-processing data...")
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('/', '_')
df.rename(columns={'nacionality':'nationality'}, inplace=True)

df['target'] = df['target'].map({
    'Dropout':0,
    'Enrolled':1,
    'Graduate':2
})


features = ['curricular_units_2nd_sem_(approved)',
       'curricular_units_2nd_sem_(grade)',
       'curricular_units_1st_sem_(approved)',
       'curricular_units_1st_sem_(grade)', 'tuition_fees_up_to_date',
       'scholarship_holder', 'age_at_enrollment', 'debtor', 'gender',
       'application_mode', 'curricular_units_2nd_sem_(enrolled)',
       'curricular_units_1st_sem_(enrolled)', 'displaced', 'target']

df = df[features]


# Train-val-test split
print(f"Performing train-val-test split...")
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=13)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=13)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.target.values
y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values

# Drop column 'target'
df_full_train.drop('target', axis=1, inplace=True)
df_train.drop('target', axis=1, inplace=True)
df_val.drop('target', axis=1, inplace=True)
df_test.drop('target', axis=1, inplace=True)

# scaling features
print(f"Scaling data...")
scaler = StandardScaler()

df_full_train_scaled = scaler.fit_transform(df_full_train)
df_train_scaled = scaler.fit_transform(df_train)
df_val_scaled = scaler.transform(df_val)  # Only transform, don't fit!
df_test_scaled = scaler.transform(df_test)  # Only transform, don't fit!

# handling class imbalance - Use class weights
print(f"handling class imbalance...")
class_weights = compute_class_weight('balanced',
                                   classes=np.unique(y_train),
                                   y=y_train)
weight_dict = dict(zip(np.unique(y_train), class_weights))


# Model training- random forest
print("-" * 50)
print(f"Training model without cross-validation...")
model_rf = RandomForestClassifier(n_estimators=200,
                                          max_depth=10,
                                          min_samples_leaf=3,
                                          class_weight=weight_dict,
                                          n_jobs=-1,
                                          random_state=13)
# Fit the model
model_rf.fit(df_train_scaled, y_train)

# Get predictions and calculate ROC-AUC
y_pred_proba = model_rf.predict_proba(df_val_scaled)

# Handle both binary and multi-class cases
if len(np.unique(y_train)) == 2:
    # Binary classification
    y_pred_proba = y_pred_proba[:, 1]  # Only need probability of positive class
    roc_auc = roc_auc_score(y_val, y_pred_proba)
else:
    # Multi-class classification
    roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='macro')

print(f"\nROC-AUC score: {roc_auc:.4f}")
print("-" * 50)


# Train with cross-validation
print(f"Training model with cross-validation...")
def custom_roc_auc(y_true, y_pred_proba):
    """Custom ROC-AUC scorer that handles both binary and multi-class cases"""
    if y_pred_proba.ndim == 1:
        return roc_auc_score(y_true, y_pred_proba)
    elif y_pred_proba.shape[1] == 2:
        return roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')

# Create a custom scorer
roc_auc_scorer = make_scorer(custom_roc_auc, response_method='predict_proba')

# Perform cross-validation
try:
    scores = cross_val_score(
        model_rf,
        df_train_scaled,
        y_train,
        cv=5,
        scoring=roc_auc_scorer,
        n_jobs=-1
    )

    # Print results with confidence interval
    print("\nCross-validation results:")
    print(f"Mean ROC-AUC: {scores.mean():.4f} ± {scores.std() * 1.96:.4f}")
    print(f"Individual fold scores: {', '.join([f'{score:.4f}' for score in scores])}")
    print("-" * 50)

except Exception as e:
    print(f"Error during cross-validation: {str(e)}")


# random forest - Train final model using train + val
print(f"Training final model using train+val...")
model_rf = RandomForestClassifier(n_estimators=200,
                                          max_depth=10,
                                          min_samples_leaf=3,
                                          class_weight=weight_dict,
                                          n_jobs=-1,
                                          random_state=13)
# Fit the model
model_rf.fit(df_full_train_scaled, y_full_train)

# Get predictions and calculate ROC-AUC
y_pred_proba = model_rf.predict_proba(df_test_scaled)

# Handle both binary and multi-class cases
if len(np.unique(y_train)) == 2:
    # Binary classification
    y_pred_proba = y_pred_proba[:, 1]  # Only need probability of positive class
    roc_auc = roc_auc_score(y_test, y_pred_proba)
else:
    # Multi-class classification
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

print(f"\nROC-AUC score: {roc_auc:.4f}")
print("-" * 50)


# Save model + scaler
print(f"Saving model...")
output_file = f"model_rf.bin"

with open(output_file, 'wb') as f_out:
    pickle.dump((scaler, model_rf), f_out)

print(f"All steps completed!")
print("-" * 50)