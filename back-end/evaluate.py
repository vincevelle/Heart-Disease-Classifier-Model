# File for evaluating the model(s)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
import joblib
import pandas as pd
import numpy as np

# -------------------------------
# Load Preprocessed Data
# -------------------------------
df = pd.read_csv("data/heart_disease_preprocessed.csv")
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# Apply Scaling to Test Data
# -------------------------------
scaler = joblib.load("data/scaler.pkl")
X_test = scaler.transform(X_test)  # Ensure consistency with training

# -------------------------------
# Evaluate Baseline Model
# -------------------------------
baseline_model = joblib.load("baseline_model.pkl")
y_pred_baseline = baseline_model.predict(X_test)
y_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

print("Baseline Model Classification Report:")
print(classification_report(y_test, y_pred_baseline))

fpr, tpr, thresholds = roc_curve(y_test, y_proba_baseline)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="Logistic Regression ROC (AUC = %0.2f)" % roc_auc, linewidth=2)
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Baseline Model")
plt.legend(loc="lower right")
plt.savefig("baseline_roc_curve.png", dpi=300)
plt.close()
print("Baseline model ROC curve saved as 'baseline_roc_curve.png'.")

# -------------------------------
# Evaluate Advanced Model (Neural Network)
# -------------------------------
from tensorflow.keras.models import load_model

advanced_model = load_model("advanced_model.h5")
y_proba_advanced = advanced_model.predict(X_test).ravel()

# Find optimal threshold using Youden's Index
fpr_adv, tpr_adv, thresholds_adv = roc_curve(y_test, y_proba_advanced)
optimal_idx = np.argmax(tpr_adv - fpr_adv)  # Youden’s Index
optimal_threshold = thresholds_adv[optimal_idx]

y_pred_advanced = (y_proba_advanced >= optimal_threshold).astype(int)
print(f"Optimal threshold for NN: {optimal_threshold:.2f}")

print("Advanced Model Classification Report:")
print(classification_report(y_test, y_pred_advanced))

roc_auc_adv = auc(fpr_adv, tpr_adv)

plt.figure(figsize=(8, 6))
plt.plot(fpr_adv, tpr_adv, label="Neural Network ROC (AUC = %0.2f)" % roc_auc_adv, linewidth=2)
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Advanced Model")
plt.legend(loc="lower right")
plt.savefig("advanced_roc_curve.png", dpi=300)
plt.close()
print("Advanced model ROC curve saved as 'advanced_roc_curve.png'.")
