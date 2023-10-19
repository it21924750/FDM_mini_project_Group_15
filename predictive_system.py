# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle  

loaded_model = pickle.load(open('C:/Users/udese\Downloads/trainedmodelsav/trained_model.sav','rb'))

input_data = (45,2,145,56,140,90,2,1)

#  change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# # Reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has a Heart Disease')