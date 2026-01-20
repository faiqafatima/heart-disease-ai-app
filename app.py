import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="🫀",
    layout="wide"
)

# 2. DEFINE THE MODEL
model = nn.Sequential(
    nn.Linear(13, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# 3. LOAD THE BRAIN
try:
    model.load_state_dict(torch.load('heart_model.pth'))
    model.eval()
except:
    st.error("❌ Error: 'heart_model.pth' not found.")

# --- SCALING FUNCTION (Crucial for correct results) ---
def preprocess_input(data):
    # Mean and Std Dev from the UCI Heart Disease dataset
    means = np.array([54.4, 0.68, 0.96, 131.6, 246.0, 0.14, 0.52, 149.6, 0.32, 1.04, 1.40, 0.72, 2.31])
    stds = np.array([9.0, 0.46, 1.03, 17.5, 51.5, 0.35, 0.52, 22.9, 0.46, 1.16, 0.61, 1.02, 0.61])
    
    input_array = np.array(data)
    scaled_data = (input_array - means) / stds
    return torch.FloatTensor([scaled_data])

# 4. APP HEADER & DISCLAIMER
st.title("🫀 HeartGuard AI: Cardiac Risk Prediction")
st.info("ℹ️ NOTE: This model is trained on clinical data for adults (Age 20-80). It is not suitable for children.")
st.markdown("---")

# 5. INPUTS (Columns)
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Demographics")
    # LOCKED: Age starts at 20
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")
    trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)

with col2:
    st.header("🩺 Vitals & Labs")
    chol = st.slider("Cholesterol", 100, 400, 200)
    fbs = st.radio("Fasting Blood Sugar > 120", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("Resting ECG Result", options=[0, 1, 2], format_func=lambda x: f"Result {x}")
    thalach = st.slider("Max Heart Rate", 70, 220, 150)

with col3:
    st.header("🫀 Heart Data")
    exang = st.radio("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.slider("Slope of Peak Exercise", 0, 2, 1)
    ca = st.slider("Major Vessels (0-4)", 0, 4, 0)
    thal = st.slider("Thalassemia (0-3)", 0, 3, 2)

# 6. PREDICTION LOGIC
st.markdown("---")
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    if st.button("🔍 Analyze Patient Risk", use_container_width=True):
        
        # 1. Collect Raw Data
        raw_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        
        # 2. Scale the Data
        input_tensor = preprocess_input(raw_data)
        
        # 3. Predict
        with torch.no_grad():
            prediction = model(input_tensor)
            risk_score = prediction.item()
        
        # 4. Display Results
        st.subheader("Analysis Result:")
        
        # Progress Bar Color Logic
        if risk_score < 0.3:
            bar_color = "green"
        elif risk_score < 0.7:
            bar_color = "yellow"
        else:
            bar_color = "red"
            
        st.progress(risk_score)
        st.metric(label="Heart Disease Probability", value=f"{risk_score:.2%}")

        if risk_score > 0.5:
            st.error(f"⚠️ HIGH RISK DETECTED (Score: {risk_score:.4f})")
            st.write("Recommendation: Consult a cardiologist immediately.")
        else:
            st.success(f"✅ LOW RISK (Score: {risk_score:.4f})")
            st.write("Patient appears healthy based on provided metrics.")