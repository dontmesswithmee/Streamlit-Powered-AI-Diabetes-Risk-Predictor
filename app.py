import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

# Load the trained model and data processing objects
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
selector = pickle.load(open('selector.pkl', 'rb'))

st.title('Diabetes Risk Prediction')

def predict_diabetes(features):
    features_scaled = scaler.transform([features])
    features_selected = selector.transform(features_scaled)
    prediction = model.predict(features_selected)
    return prediction

st.sidebar.header('Input Features')
pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 0)
glucose = st.sidebar.number_input('Glucose', 0, 200, 0)
blood_pressure = st.sidebar.number_input('Blood Pressure', 0, 150, 0)
skin_thickness = st.sidebar.number_input('Skin Thickness', 0, 100, 0)
insulin = st.sidebar.number_input('Insulin', 0, 900, 0)
bmi = st.sidebar.number_input('BMI', 0.0, 70.0, 0.0)
dpf = st.sidebar.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.0)
age = st.sidebar.number_input('Age', 0, 120, 0)

features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

if st.sidebar.button('Predict'):
    prediction = predict_diabetes(features)
    if prediction == 1:
        st.write("You have a high risk of diabetes.")
    else:
        st.write("You have a low risk of diabetes.")
