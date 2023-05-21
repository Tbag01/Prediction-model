# -*- coding: utf-8 -*-
"""
Created on Sun May 21 18:56:25 2023

@author: DELL
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
lr_loaded = pickle.load(open('C:/Users/DELL/Desktop/School Work/Part 4.2/Dissertation/Model Deployment/linear_regression_model.pkl', 'rb'))

def insurance_prediction(input_data):
    #num_cols = ['age_of_license', 'speed', 'distance', 'time_spent_driving', 'number_of_previous_claims', 'time_of_last_claim', 'car_model_year', 'age']
    input_data = (10, 180, 300, 6, 3, 5, 1, 26)

    input_data_reshaped = np.array([10, 180, 300, 6, 3, 5, 1, 26]).reshape(1, -1)

    prediction = lr_loaded.predict(input_data_reshaped)

    print('The insurance premium is: ', prediction)
    
    
def main():
        
    #giving a title
    st.title('Insurance Prediction Web App')
    
    #getting the input data from the user
    age_of_license = st.text_input('Age of license')
    speed = st.text_input('Speed')
    distance = st.text_input('Distance')
    time_spent_driving = st.text_input('Time spent driving')
    number_of_previous_claims = st.text_input('Number of previous claims')
    time_of_last_claim = st.text_input('Number of years last made a claim')
    car_model_year = st.text_input('Car model year')
    age = st.text_input('Age of driver')
    
    #code for prediction
    insurance = ''
    
    #creating a button fro prediction
    if st.button('Prediction result'):
        insurance = insurance_prediction(age_of_license, speed, distance, time_spent_driving, number_of_previous_claims, time_of_last_claim, car_model_year, age)
        
    st.success(insurance)
    
    
if __name__== '__main__':
    main()