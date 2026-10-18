import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from scikeras.wrappers import KerasClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    LeakyReLU,
)


# -------------------------------
# Configuration
# -------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SEED = 42

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# -------------------------------
# Neural Network Architecture
# -------------------------------

def create_nn_model(meta, dropout_rate=0.2, optimizer="adam"):
    model = Sequential([
        Input(shape=(meta["n_features_in_"],)),

        Dense(64, kernel_initializer="he_uniform"),
        LeakyReLU(negative_slope=0.1),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(32, kernel_initializer="he_uniform"),
        LeakyReLU(negative_slope=0.1),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(16, kernel_initializer="he_uniform"),
        LeakyReLU(negative_slope=0.1),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# -------------------------------
# Neural Network ROC AUC Scorer
# -------------------------------

def nn_auc_scorer(estimator, X, y):
    """
    Calculate binary ROC AUC using the probability
    of the positive class (HeartDisease = 1).
    """
    probabilities = np.asarray(estimator.predict_proba(X))

    if probabilities.ndim == 2:
        probabilities = probabilities[:, 1]

    return roc_auc_score(y, probabilities)


def positive_class_probabilities(probabilities):
    """
    Convert binary classifier probability output
    into a one-dimensional array.
    """
    probabilities = np.asarray(probabilities)

    if probabilities.ndim == 2:
        return probabilities[:, 1]

    return probabilities.ravel()


# -------------------------------
# Main Training Pipeline
# -------------------------------

def main():

    # -------------------------------
    # Load Preprocessed Data
    # -------------------------------

    df = pd.read_csv(
        DATA_DIR / "heart_disease_preprocessed.csv"
    )

    X = df.drop(columns="HeartDisease")
    y = df["HeartDisease"]

    # Split BEFORE fitting any imputer or scaler
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    # Cross-validation configurations
    cv_lr = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    cv_nn = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    # -------------------------------
    # Baseline: Logistic Regression
    # -------------------------------

    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=SEED,
            ),
        ),
    ])

    lr_search = RandomizedSearchCV(
        estimator=baseline,
        param_distributions={
            "model__C": [0.01, 0.1, 1, 10, 100],
        },
        n_iter=5,
        cv=cv_lr,
        scoring="roc_auc",
        random_state=SEED,
        n_jobs=-1,
    )

    lr_search.fit(X_train, y_train)

    print(
        "Best logistic regression parameters:",
        lr_search.best_params_,
    )

    joblib.dump(
        lr_search.best_estimator_,
        ROOT / "baseline_model.pkl",
    )

    print("Baseline model saved.")

    # -------------------------------
    # Advanced Model: Neural Network
    # -------------------------------

    nn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            KerasClassifier(
                model=create_nn_model,
                verbose=0,
                random_state=SEED,
                fit__validation_split=0.2,
                callbacks=[EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                )],
            ),
        ),
    ])

    # Hyperparameter search space
    param_grid = {
        "model__batch_size": [16, 32, 64],
        "model__epochs": [50, 100, 150],
        "model__model__optimizer": ["adam", "rmsprop"],
        "model__model__dropout_rate": [0.2, 0.3, 0.4],
    }

    nn_search = RandomizedSearchCV(
        estimator=nn,
        param_distributions=param_grid,
        cv=cv_nn,
        scoring=nn_auc_scorer,
        n_iter=10,
        random_state=SEED,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )

    nn_search.fit(X_train, y_train)

    print(
        "Best neural network parameters:",
        nn_search.best_params_,
    )

    best_nn = nn_search.best_estimator_

    # Plot the loss history of the final refitted neural network.
    history = best_nn.named_steps["model"].history_
    plt.figure(figsize=(8, 6))
    plt.plot(history["loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Neural Network Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    loss_plot_path = DATA_DIR / "advanced_loss_curve.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Saved {loss_plot_path}")

    # -------------------------------
    # Optimize Classification Threshold
    # Using Training Data Only
    # -------------------------------

    # Obtain out-of-fold predictions.
    # Each fold fits its own imputer, scaler, and model.
    oof_probabilities = positive_class_probabilities(
        cross_val_predict(
            best_nn,
            X_train,
            y_train,
            cv=cv_nn,
            method="predict_proba",
            n_jobs=1,
        )
    )

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(
        y_train,
        oof_probabilities,
    )

    # Youden's Index = Sensitivity + Specificity - 1
    # Equivalent to TPR - FPR
    finite = np.isfinite(thresholds)

    optimal_idx = np.argmax(
        np.where(finite, tpr - fpr, -np.inf)
    )

    threshold = float(thresholds[optimal_idx])

    print(
        f"Training-derived neural network threshold: "
        f"{threshold:.4f}"
    )

    # -------------------------------
    # Save Advanced Model
    # -------------------------------

    best_nn.named_steps["model"].model_.save(
        ROOT / "advanced_model.keras"
    )

    # Save fitted imputer, scaler, feature order,
    # and classification threshold.
    joblib.dump(
        {
            "imputer": best_nn.named_steps["imputer"],
            "scaler": best_nn.named_steps["scaler"],
            "feature_names": X_train.columns.tolist(),
            "threshold": threshold,
        },
        DATA_DIR / "advanced_preprocessing.pkl",
    )

    print("Advanced model saved.")
    print(
        "Saved baseline_model.pkl, "
        "advanced_model.keras, and "
        "data/advanced_preprocessing.pkl"
    )


if __name__ == "__main__":
    main()
