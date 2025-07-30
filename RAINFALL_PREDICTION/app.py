import streamlit as st
import pickle
import numpy as np
import base64

# Function to set background image
def set_local_bg(image_file):
    with open(image_file, "rb") as img:
        b64_string = base64.b64encode(img.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{b64_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set background
set_local_bg("images.jpg")

# Load the model and feature names
with open("University_ML_project.pkl", "rb") as file:
    data = pickle.load(file)
    model = data["model"]
    feature_names = data["feature_names"]

# App title and instructions
st.title("🌧️ Rainfall Prediction App")
st.markdown("### Enter the following weather data:")

# Reset state check
if "reset" not in st.session_state:
    st.session_state.reset = False

# Input section
user_inputs = []
for feature in feature_names:
    key = f"input_{feature}"
    if st.session_state.reset:
        st.session_state[key] = 0.0
    value = st.number_input(f"{feature}", key=key, step=0.1, format="%.2f")
    user_inputs.append(value)

# Predict button
if st.button("🔍 Predict Rainfall"):
    input_array = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    if prediction == 1:
        st.success("🌧️ Rain is likely tomorrow.")
    else:
        st.info("☀️ No rain is expected tomorrow.")

# Refresh button
if st.button("🔄 Refresh Inputs"):
    st.session_state.reset = True
    st.rerun()  # Updated method for rerun

# Clear reset flag after rerun
if st.session_state.reset:
    st.session_state.reset = False
