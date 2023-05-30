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

def insurance_prediction(age_of_license, speed, distance, time_spent_driving, number_of_previous_claims, time_of_last_claim, car_model_year, age):
    #num_cols = ['age_of_license', 'speed', 'distance', 'time_spent_driving', 'number_of_previous_claims', 'time_of_last_claim', 'car_model_year', 'age']
    input_data = (age_of_license, speed, distance, time_spent_driving, number_of_previous_claims, time_of_last_claim, car_model_year, age)

    input_data_reshaped = np.array(input_data).reshape(1, -1) 

    prediction = lr_loaded.predict(input_data_reshaped)

    return prediction
    
    
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
    
    #creating a button for prediction
    if st.button('Prediction result'):
        insurance = insurance_prediction(int(age_of_license), 
                                 float(speed), 
                                 float(distance), 
                                 int(time_spent_driving), 
                                 int(number_of_previous_claims), 
                                 int(time_of_last_claim), 
                                 int(car_model_year), 
                                 int(age))
        
    st.success(insurance)
    
    
if __name__== '__main__':
    main()