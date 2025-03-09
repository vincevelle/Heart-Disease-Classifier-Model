# File for training models

# models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# For advanced model (Neural Network)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from scikeras.wrappers import KerasClassifier

# -------------------------------
# Load Preprocessed Data
# -------------------------------
df = pd.read_csv('data/heart_disease_preprocessed.csv')

# Separate features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Split data (80/20 split, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


scaler = joblib.load("data/scaler.pkl")

# Compare feature names
print("Features from saved scaler:")
print(scaler.feature_names_in_)  # Requires scikit-learn 1.1+

print("\nFeatures in the current dataset:")
print(X_train.columns)

# -------------------------------
# Load Pre-Saved Scaler and Apply Scaling
# -------------------------------
scaler = joblib.load("data/scaler.pkl")
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Baseline Model: Logistic Regression
# -------------------------------
def train_logistic_regression(X_train, y_train):
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    lr = LogisticRegression(max_iter=1000, solver='liblinear')
    grid = RandomizedSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_iter=5, random_state=42)
    grid.fit(X_train, y_train)
    print("Best parameters for Logistic Regression:", grid.best_params_)
    return grid.best_estimator_

baseline_model = train_logistic_regression(X_train, y_train)
joblib.dump(baseline_model, "baseline_model.pkl")
print("Baseline model saved as 'baseline_model.pkl'.")

# -------------------------------
# Advanced Model: Neural Network
# -------------------------------
def create_nn_model(optimizer='adam', dropout_rate=0.2, **kwargs):
    model = Sequential()
    model.add(Dense(64, kernel_initializer='he_uniform', input_dim=X_train.shape[1]))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(16, kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap the Keras model for use in scikit-learn
nn_model = KerasClassifier(model=create_nn_model, verbose=0, classifier=True)

# Set up hyperparameter grid using RandomizedSearchCV for efficiency
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 150],
    'optimizer': ['adam', 'rmsprop'],
    'model__dropout_rate': [0.2, 0.3, 0.4]
}

grid_nn = RandomizedSearchCV(estimator=nn_model, param_distributions=param_grid, 
                             cv=3, scoring='roc_auc', n_iter=10, random_state=42)
grid_nn.fit(X_train, y_train)
print("Best parameters for Neural Network:", grid_nn.best_params_)

advanced_model = grid_nn.best_estimator_.model_
advanced_model.save("advanced_model.h5")
print("Advanced Neural Network model saved as 'advanced_model.h5'.")


