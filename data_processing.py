# File for data processing
# data_processing.py

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load the raw dataset
# -------------------------------

df = pd.read_csv('heart_disease.csv')

print("Initial Data Information:")
print(df.info())
print("First few rows:")
print(df.head())

# -------------------------------
# Data Cleaning
# -------------------------------

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# -------------------------------
# Feature Selection
# -------------------------------

selected_features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
                     'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease']
df = df[selected_features]

# -------------------------------
# Categorical Encoding
# -------------------------------

df = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)

# -------------------------------
# Feature Engineering
# -------------------------------

# Combining strong predictors to capture possible joint effects
df['Oldpeak_x_ExerciseAngina'] = df['Oldpeak'] * df['ExerciseAngina_Y']
df['MaxHR_x_Age'] = df['MaxHR'] * df['Age']
df['ST_Slope_Flat_x_Oldpeak'] = df['ST_Slope_Flat'] * df['Oldpeak']

# -------------------------------
# Normalization/Standardization
# -------------------------------
X = df.drop("HeartDisease", axis=1)
scaler = StandardScaler()
scaler.fit(X)

# numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save the scaler for later use
scaler_path = "data/scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

# -------------------------------
# Save Preprocessed Data
# -------------------------------
preprocessed_data_path = "data/heart_disease_preprocessed.csv"

for col in df.select_dtypes(include=['bool']).columns:
    df[col] = df[col].astype(int)

df.to_csv(preprocessed_data_path, index=False)
print(f"Preprocessed data saved to {preprocessed_data_path}.")

# -------------------------------
# Exploratory Data Analysis (EDA)
# -------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()

# Save correlation matrix plot
correlation_matrix_path = "data/correlation_matrix.png"
plt.savefig(correlation_matrix_path)
plt.close()

print(f"Correlation matrix saved as {correlation_matrix_path}.")
