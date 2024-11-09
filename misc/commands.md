## Dataset:

1. https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
2. https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data


## Project structure

.
├── data
│   └── dataset.csv
├── notebooks
│   ├── model-train.ipynb
│   ├── model_rf.bin
│   └── notebook.ipynb
├── Dockerfile
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── README.md
├── model_rf.bin
├── predict-local-test.py
├── predict-requests-test.py
├── predict.py
└── train.py

## Setup

```python
pip install pipenv

pipenv install ipykernel pandas scikit-learn xgboost tqdm flask gunicorn

pipenv install --dev notebook ipywidgets matplotlib seaborn

pipenv run python -m ipykernel install --user --name=student-dropout-and-success-prediction --display-name "Python (student-dropout-and-success-prediction)"
```

Verification:
```
pipenv run pip show xgboost
```

Run jupyter-lab:
```
pipenv run jupyter-lab
```

## Train and test locally

1. Train and save model

```
pipenv run python train.py
```
Result:
```
Importing required libraries...
Loading data...
Pre-processing data...
Performing train-val-test split...
Scaling data...
handling class imbalance...
--------------------------------------------------
Training model without cross-validation...

ROC-AUC score: 0.8661
--------------------------------------------------
Training model with cross-validation...

Cross-validation results:
Mean ROC-AUC: 0.8763 ± 0.0193
Individual fold scores: 0.8813, 0.8819, 0.8567, 0.8825, 0.8792
--------------------------------------------------
Training final model using train+val...

ROC-AUC score: 0.8843
--------------------------------------------------
Saving model...
All steps completed!
--------------------------------------------------
```

2. Load saved model and test

```
pipenv run python predict-local-test.py
```

Result:
```
Importing required libraries...
Loading model...
input: {'curricular_units_2nd_sem_(approved)': [8], 'curricular_units_2nd_sem_(grade)': [14.07125], 'curricular_units_1st_sem_(approved)': [8], 'curricular_units_1st_sem_(grade)': [14.07125], 'tuition_fees_up_to_date': [1], 'scholarship_holder': [1], 'age_at_enrollment': [19], 'debtor': [0], 'gender': [0], 'application_mode': [1], 'curricular_units_2nd_sem_(enrolled)': [8], 'curricular_units_1st_sem_(enrolled)': [8], 'displaced': [1]}
--------------------------------------------------
Student 1: Prediction = Graduate, Probabilities = [0.02719054 0.09465016 0.8781593 ]
```

## Running flask app and test

```
pipenv run python predict.py
```

#### Testing From another terminal, run:
```
pipenv run python predict-requests-test.py
```

Result:
```
input: {'curricular_units_2nd_sem_(approved)': [8], 'curricular_units_2nd_sem_(grade)': [14.07125], 'curricular_units_1st_sem_(approved)': [8], 'curricular_units_1st_sem_(grade)': [14.07125], 'tuition_fees_up_to_date': [1], 'scholarship_holder': [1], 'age_at_enrollment': [19], 'debtor': [0], 'gender': [0], 'application_mode': [1], 'curricular_units_2nd_sem_(enrolled)': [8], 'curricular_units_1st_sem_(enrolled)': [8], 'displaced': [1]}
--------------------------------------------------
{'predictions by model': [{'predicted_status': 'Graduate', 'probabilities': {'Dropout': 0.027190541365371718, 'Enrolled': 0.09465016268419602, 'Graduate': 0.8781592959504326}, 'student_id': 1}]}
```

## Running flask app via gunicorn
```
pipenv shell
gunicorn --bind 0.0.0.0:9696 predict:app
```
OR
```
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```

#### Testing From another terminal, run:
```
pipenv run python predict-requests-test.py
```

Result: Same as previous step


## Containerization

1. Build and run docker image (for system dependency management)

```
docker build -t student-dropout-success .
docker images
```
2. Run the container with proper signal handling
```
docker run -it --rm \
  --name student-dropout-container \
  -p 9696:9696 \
  --stop-signal SIGTERM \
  --stop-timeout 30 \
  student-dropout-success
```
You can stop the container gracefully by:

- Using Ctrl+C if running in interactive mode
- Running `docker stop student-dropout-container` from another terminal

3. Test from another terminal

```
python predict-requests-test.py
```
Result: Same as previous step


## Cloud Deployment
