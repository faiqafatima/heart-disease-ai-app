import torch
import torch.nn as nn
import numpy as np

# 1. DEFINE THE ARCHITECTURE (Must match training!)
# We had 13 inputs -> 32 hidden -> 1 output
model = nn.Sequential(
    nn.Linear(13, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# 2. LOAD THE WEIGHTS
# We load the "brain" we trained on Kaggle into this empty structure
try:
    model.load_state_dict(torch.load('heart_model.pth'))
    model.eval() # Set to evaluation mode (important!)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'heart_model.pth' not found. Did you download it?")
    exit()

# 3. LOAD PATIENT DATA
try:
    with open('test_patient.txt', 'r') as f:
        data_str = f.read().strip()
        # Convert "57, 1, 0..." string into a list of numbers
        patient_data = [float(x) for x in data_str.split(',')]
        
    # Convert to PyTorch Tensor (1 row, 13 columns)
    patient_tensor = torch.FloatTensor([patient_data])
    
    print(f"🏥 Analyzing Patient Data: {patient_data}")

except Exception as e:
    print(f"❌ Error reading text file: {e}")
    exit()

# 4. PREDICT
with torch.no_grad(): # No training needed, just math
    prediction = model(patient_tensor)
    risk_score = prediction.item() # Convert tensor to python number

# 5. RESULT
print("-" * 30)
print(f"❤️ HEART DISEASE RISK SCORE: {risk_score:.4f}")
print("-" * 30)

if risk_score > 0.5:
    print("⚠️ HIGH RISK: Detection of Heart Disease likely.")
else:
    print("✅ LOW RISK: Patient looks healthy.")