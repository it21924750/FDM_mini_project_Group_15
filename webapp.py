# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:18:14 2023

@author: udese
"""


import numpy as np
import pickle
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

#loading the saved model
loaded_model = pickle.load(open('C:/Users/udese/Downloads/trained_model.sav', 'rb')) 


#Creating a function for prediction
def heart_prediction(modified_input_data):
    
    
    # Unpack the tuple into individual variables
    age_in_days, weight, ap_hi, ap_lo, cholesterol, pp, bmi, health_risk_score  = modified_input_data

    #Change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(modified_input_data)

    # Reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    #Select the model
    selected_model = loaded_model

    prediction = selected_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
      return'The Person does not have a Heart Disease'
    else:
      return'The Person has a Heart Disease'



        
def main():
    
    # Giving a title 
    st.title("Heart Disease Prediction Web App")
    
    # Getting the input data from the user
    
    #for dropdowns
    
    # Map numerical values to labels
    gender_mapping = {1: 'Male', 2: 'Female'}
    cholestrol_mapping = {1: 'normal', 2: 'above normal', 3: 'well above normal'}
    gluc_mapping = {1: 'normal', 2: 'above normal', 3: 'well above normal'}
    smoke_mapping = {1:'smoker', 0:'non-smoker'}
    alco_mapping = {1:'yes', 0:'no'}
    active_mapping = {1:'active', 0:'inactive'}
    
    # Create a list of numerical values from your dataset
    gender_values = [1, 2]
    cholestrol_values = [1,2,3]
    gluc_values = [1,2,3]
    smoke_values = [0,1]
    alco_values = [0,1]
    active_values = [0,1]
    
    # Use a list comprehension to create a list of corresponding labels
    gender_options = [gender_mapping[value] for value in gender_values]
    cholestrol_options = [cholestrol_mapping[value] for value in cholestrol_values]
    gluc_options = [gluc_mapping[value] for value in gluc_values]
    smoke_options = [smoke_mapping[value] for value in smoke_values]
    alco_options = [alco_mapping[value] for value in alco_values]
    active_options = [active_mapping[value] for value in active_values]
    
    # age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active
    
    #age
    age_str = st.text_input('Age in years')
    age = int(age_str) if age_str else 0  # Default to 0 if input is empty


    #gender
    # Use st.selectbox to create a dropdown menu for gender
    selected_gender_label = st.selectbox('Select Gender', gender_options)
    # Reverse map the selected label to the numerical value
    gender = {label: value for value, label in gender_mapping.items()}[selected_gender_label]
    # Display the selected gender (numerical value)
    st.write(f'You selected gender: {selected_gender_label} (numerical value: {gender})')

    #height
    height_str = st.text_input('Height measured in centimeters ')
    height = int(height_str) if height_str else 0  # Default to 0 if input is empty

    #weight
    weight_str = st.text_input('Weight measured in kilograms ')
    weight = int(weight_str) if weight_str else 0  # Default to 0 if input is empty

    #ap_hi
    ap_hi_str = st.text_input('Systolic blood pressure ')
    ap_hi = int(ap_hi_str) if ap_hi_str else 0  # Default to 0 if input is empty

    #ap_lo
    ap_lo_str = st.text_input('Diastolic blood pressure ')
    ap_lo = int(ap_lo_str) if ap_lo_str else 0  # Default to 0 if input is empty

    #Cholestrol
    selected_cholestrol_label = st.selectbox('Select cholestrol Level', cholestrol_options)
    cholestrol = {label: value for value, label in cholestrol_mapping.items()}[selected_cholestrol_label]
    st.write(f'selected cholestrol level: {selected_cholestrol_label} (numerical value: {cholestrol})')

    #Glucose
    selected_gluc_label = st.selectbox('Select glucose Level', gluc_options)
    gluc = {label: value for value, label in gluc_mapping.items()}[selected_gluc_label]
    st.write(f'selected gluc level: {selected_gluc_label} (numerical value: {gluc})')

    #smoke
    selected_smoke_label = st.selectbox('Do you smoke? ', smoke_options)
    smoke = {label: value for value, label in smoke_mapping.items()}[selected_smoke_label]
    st.write(f'Smokes or not: {selected_smoke_label} (numerical value: {smoke})')

    #alco
    selected_alco_label = st.selectbox('Do you consume Alcohol? ', alco_options)
    alco = {label: value for value, label in alco_mapping.items()}[selected_alco_label]
    st.write(f'Drinks alcohol or not: {selected_alco_label} (numerical value: {alco})')

    #active
    selected_active_label = st.selectbox('Are you physically active? ', active_options)
    active = {label: value for value, label in active_mapping.items()}[selected_active_label]
    st.write(f'Physically active or not: {selected_active_label} (numerical value: {active})')

    # Code for prediction
    
    #return variable will be stored here
    diagnosis = ''
    

    #create a button
    if st.button('Heart Disease Test Result'):
        # Pass the input data to the function as a tuple
        input_data = (
            age,
            gender,
            height,
            weight,
            ap_hi,
            ap_lo,
            cholestrol,
            gluc,
            smoke,
            alco,
            active
        )
        
        # Unpack the tuple into individual variables
        age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active = input_data

        # Convert age from years to days
        age_in_days = age * 365  # Assuming an average year has 365 days

        # BMI and pulse pressure
        pp = ap_hi - ap_lo
        bmi = weight / ((height / 100) * (height / 100))

        # Calculating health risk score
        weights = {
            'Chol': 3,
            'Smoke': 3,
            'Gluc': 2,
            'Alco': 1
        }

        health_risk_score = cholesterol * weights['Chol'] + gluc * weights['Gluc'] + smoke * weights['Smoke'] + alco * weights['Alco']

        # Create modified input data as a tuple
        modified_input_data = (age_in_days, weight, ap_hi, ap_lo, cholesterol, pp, bmi, health_risk_score)
        
        # call the function
        diagnosis = heart_prediction(modified_input_data)
        
        
    st.success(diagnosis)   
            
    # Display the plot here
    if diagnosis:  # Display plot only if there is a diagnosis
    
        # Create a DataFrame for the user's data
        user_df = pd.DataFrame([modified_input_data], index=['Value'], columns=['age', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'pp','bmi','health_risk_score'])

        # Initialize an explainer with the loaded model
        explainer = shap.Explainer(loaded_model)

        # Calculate SHAP values for the user's data
        shap_values = explainer.shap_values(user_df)

        # Plot the summary plot
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, user_df, plot_type="bar", show=False)
        plt.title("Feature Importance for Heart Disease Prediction")
        st.pyplot(fig, clear_figure=True)  # Display the plot in Streamlit app
            


 
#Run the file using a command prompt or as a standalone file
if __name__ == '__main__':
    main()
    
        