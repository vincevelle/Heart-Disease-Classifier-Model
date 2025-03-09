# File for building the web app
# app.py

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder="front-end/templates")


# -------------------------------
# Load the Advanced Model and Scaler
# -------------------------------
model = load_model("back-end/advanced_model.h5")
with open("back-end/data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# Define Input Fields (Must Match Preprocessed Data)
# -------------------------------
input_fields = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
                'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
                'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
                'ST_Slope_Flat', 'ST_Slope_Up', 'Oldpeak_x_ExerciseAngina',
                'MaxHR_x_Age', 'ST_Slope_Flat_x_Oldpeak']

@app.route('/')
def home():
    return render_template('home.html', fields=['Age', 'Sex (M/F)', 'RestingBP', 'Cholesterol', 
                                                 'FastingBS', 'MaxHR', 'Oldpeak', 
                                                 'ChestPainType (ATA/NAP/TA)', 
                                                 'RestingECG (Normal/ST)', 'ExerciseAngina (0/1)',
                                                 'ST_Slope (Flat/Up)'])

def preprocess_input(form_data):
    """Convert form data into a properly formatted DataFrame"""
    input_dict = {}

    # Convert numeric fields
    for field in ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']:
        input_dict[field] = float(form_data[field])

    # Encode categorical variables
    input_dict['Sex_M'] = 1 if form_data['Sex (M/F)'].lower() == 'm' else 0
    input_dict['ExerciseAngina_Y'] = 1 if form_data['ExerciseAngina (0/1)'] == '1' else 0

    # One-hot encode ChestPainType
    input_dict['ChestPainType_ATA'] = 1 if form_data['ChestPainType'].upper() == 'ATA' else 0
    input_dict['ChestPainType_NAP'] = 1 if form_data['ChestPainType'].upper() == 'NAP' else 0
    input_dict['ChestPainType_TA'] = 1 if form_data['ChestPainType'].upper() == 'TA' else 0

    # One-hot encode RestingECG
    input_dict['RestingECG_Normal'] = 1 if form_data['RestingECG'].upper() == 'NORMAL' else 0
    input_dict['RestingECG_ST'] = 1 if form_data['RestingECG'].upper() == 'ST' else 0

    # One-hot encode ST_Slope
    input_dict['ST_Slope_Flat'] = 1 if form_data['ST_Slope'].upper() == 'FLAT' else 0
    input_dict['ST_Slope_Up'] = 1 if form_data['ST_Slope'].upper() == 'UP' else 0

    # Interaction Terms
    input_dict['Oldpeak_x_ExerciseAngina'] = input_dict['Oldpeak'] * input_dict['ExerciseAngina_Y']
    input_dict['MaxHR_x_Age'] = input_dict['MaxHR'] * input_dict['Age']
    input_dict['ST_Slope_Flat_x_Oldpeak'] = input_dict['ST_Slope_Flat'] * input_dict['Oldpeak']

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure correct column order & missing one-hot encodings are filled with 0
    input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    return input_df

@app.route('/predict', methods=['POST'])
def predict():

    print("Received Form Data:", request.form)  # Debugging
    try:
        input_df = preprocess_input(request.form)
    except KeyError as e:
        return f"Missing field: {e}. Please check your form input names."

    try:
        input_df = preprocess_input(request.form)
    except ValueError:
        return "Invalid input. Please enter valid numeric and categorical values."

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict using the model
    proba = model.predict(input_scaled).ravel()[0]

    # Optimal threshold from evaluation
    optimal_threshold = 0.69
    prediction = int(proba >= optimal_threshold)

    # Determine risk level
    if proba < 0.3:
        risk = "Low risk"
    elif proba < 0.7:
        risk = "Moderate risk"
    else:
        risk = "High risk"

    result = {
        'prediction': "Presence of Heart Disease" if prediction == 1 else "No Heart Disease",
        'probability': f"{proba * 100:.2f}%",
        'risk': risk
    }

    return render_template('display.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
