#!/usr/bin/env python
# coding: utf-8

import requests

# Update below with your render web service URL
# url = "https://<your-render-url>/predict"
url = "https://student-dropout-and-success-prediction.onrender.com/predict"


# Test with 1 student
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

response = requests.post(url, json=student_data).json()


print("input:", student_data)
print("-" * 50)

print(response)
print("-" * 50)
