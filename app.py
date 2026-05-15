import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("student_performance_model.pkl")    

st.title("Student Performance Prediction")

st.subheader("Enter Student Details")
a = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, step=0.1)
b = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, step=0.1)
c = st.selectbox("Extracurricular Activities", [1, 0])
d = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, step=0.1)
e = st.number_input("Sample Question Papers Practiced", min_value=0.0, max_value=100.0, step=1.0)    
                    
if st.button("Predict Performance"):
    input_data = pd.DataFrame([[a, b, c, d, e]], columns=["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"])
    prediction = model.predict(input_data)
    st.subheader(f"Predicted Performance Score: {prediction[0]:.2f}")   

    st.success("Prediction made successfully!") 

