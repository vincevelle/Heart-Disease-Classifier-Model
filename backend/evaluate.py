"""Evaluate both models once on a held-out test set."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, roc_auc_score, classification_report,
    confusion_matrix, average_precision_score, brier_score_loss,
    balanced_accuracy_score,
)
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'


def save_roc(y_true, probabilities, title, label, path):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    score = roc_auc_score(y_true, probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{label} ROC (AUC = {score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f'{label} test AUC: {score:.4f}; saved {path}')


def print_metrics(y_true, probabilities, predictions, label):
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) else float('nan')
    print(f'{label} classification report:')
    print(classification_report(y_true, predictions, zero_division=0))
    print(f'{label} confusion matrix (rows=true, columns=predicted):')
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))
    print(f'Sensitivity: {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Balanced accuracy: {balanced_accuracy_score(y_true, predictions):.4f}')
    print(f'ROC AUC: {roc_auc_score(y_true, probabilities):.4f}')
    print(f'Average precision (PR summary): {average_precision_score(y_true, probabilities):.4f}')
    print(f'Brier score: {brier_score_loss(y_true, probabilities):.4f}')


def main():
    df = pd.read_csv(DATA_DIR / 'heart_disease_preprocessed.csv')
    X = df.drop(columns='HeartDisease')
    y = df['HeartDisease']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    baseline = joblib.load(ROOT / 'baseline_model.pkl')
    baseline_prob = baseline.predict_proba(X_test)[:, 1]
    print_metrics(y_test, baseline_prob, baseline.predict(X_test), 'Baseline')
    save_roc(y_test, baseline_prob, 'ROC Curve - Baseline Model',
             'Logistic Regression', ROOT / 'baseline_roc_curve.png')

    bundle = joblib.load(DATA_DIR / 'advanced_preprocessing.pkl')
    X_test = X_test.loc[:, bundle['feature_names']]
    X_test_imputed = bundle['imputer'].transform(X_test)
    X_test_scaled = bundle['scaler'].transform(X_test_imputed)
    advanced = load_model(ROOT / 'advanced_model.keras')
    advanced_prob = advanced.predict(X_test_scaled, verbose=0).ravel()
    advanced_pred = (advanced_prob >= bundle['threshold']).astype(int)
    print(f"Neural network training-derived threshold: {bundle['threshold']:.4f}")
    print_metrics(y_test, advanced_prob, advanced_pred, 'Advanced')
    save_roc(y_test, advanced_prob, 'ROC Curve - Advanced Model',
             'Neural Network', ROOT / 'advanced_roc_curve.png')


if __name__ == '__main__':
    main()
