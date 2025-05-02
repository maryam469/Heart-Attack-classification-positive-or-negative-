import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle


model = tf.keras.models.load_model('model.h5')

with open('label_encoder_result.pkl', 'rb') as file:
    label_encoder_result = pickle.load(file)

with open('scaler.pkl', 'rb')as file:
    scaler = pickle.load(file)

st.title("Dignosis Heart Attack Disease")

##Users input
age = st.slider("Age",18,100)
gender = st.selectbox("Gender", [1,0])
heart_rate = st.number_input("Heart rate (bpm)", min_value=30, max_value=200)
systolic = st.number_input("Systolic blood pressure (mmHg)", min_value=70, max_value=250)
diastolic = st.number_input("Diastolic blood pressure (mmHg)", min_value=40, max_value=150)
blood_sugar = st.number_input("Blood sugar (mg/dL)", min_value=50, max_value=300, value=100, help="Normal: Fasting 70–99, Random <140, Post-meal <140")
Creatine_Kinase_MB =st.number_input("CK-MB(ng/mL)", min_value=0.0, max_value=100.0)
troponin = st.number_input("Troponin(ng/mL)", min_value=0.0, max_value=10.0, step=0.01)

input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    "Heart rate" :[heart_rate],
    "Systolic blood pressure": [systolic],
    "Diastolic blood pressure":[diastolic],
    "Blood sugar": [blood_sugar],
    "CK-MB": [Creatine_Kinase_MB],
    "Troponin": [troponin],
})

input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
prediction_proba =prediction[0][0]

st.write(f"Heart Attack: {prediction_proba:.2f}")


if prediction_proba == 1:
    st.write("Positive")
else:
    st.write( "Negative") 

