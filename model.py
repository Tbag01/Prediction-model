# -*- coding: utf-8 -*-
"""
Created on Sun May 21 17:45:38 2023

@author: DELL
"""

import numpy as np
import pickle

#loading the saved model
lr_loaded = pickle.load(open('C:/Users/DELL/Desktop/School Work/Part 4.2/Dissertation/Model Deployment/linear_regression_model.pkl', 'rb'))

#num_cols = ['age_of_license', 'speed', 'distance', 'time_spent_driving', 'number_of_previous_claims', 'time_of_last_claim', 'car_model_year', 'age']
input_data = (10, 180, 300, 6, 3, 5, 1, 26)

input_data_reshaped = np.array([10, 180, 300, 6, 3, 5, 1, 26]).reshape(1, -1)

prediction = lr_loaded.predict(input_data_reshaped)

print(prediction)