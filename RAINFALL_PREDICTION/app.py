import streamlit as st
import pickle
import numpy as np

 
with open("University_ML_project.pkl", "rb") as file:
    data = pickle.load(file)
    model = data["model"]
    feature_names = data["feature_names"]

st.title("🌧️ Rainfall Prediction App")
st.markdown("### Enter the following weather data:")
 
if "reset" not in st.session_state:
    st.session_state.reset = False
 
user_inputs = []
for feature in feature_names:
    key = f"input_{feature}"
    if st.session_state.reset:
        st.session_state[key] = 0.0   
    value = st.number_input(f"{feature}", key=key, step=0.1, format="%.2f")
    user_inputs.append(value)
 
if st.button("🔍 Predict Rainfall"):
    input_array = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    if prediction == 1:
        st.success("🌧️ Rain is likely tomorrow.")
    else:
        st.info("☀️ No rain is expected tomorrow.")
 
if st.button("🔄 Refresh Inputs"):
    st.session_state.reset = True
    st.rerun()  # triggers rerun with reset values

 
if st.session_state.reset:
    st.session_state.reset = False
