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
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=13)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=13)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values

# Drop column 'target'
df_train.drop('target', axis=1, inplace=True)
df_val.drop('target', axis=1, inplace=True)
df_test.drop('target', axis=1, inplace=True)

# scaling features
print(f"Scaling data...")
scaler = StandardScaler()

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
print(f"Training model...")
model_rf = RandomForestClassifier(n_estimators=200,
                                          max_depth=10,
                                          min_samples_leaf=3,
                                          class_weight=weight_dict,
                                          n_jobs=-1,
                                          random_state=13)
model_rf.fit(df_train_scaled, y_train)

# Save model
print(f"Saving model...")
output_file = f"model_rf.bin"

with open(output_file, 'wb') as f_out:
    pickle.dump((scaler, model_rf), f_out)

print(f"All steps completed!")