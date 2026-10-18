"""Flask web interface for an educational heart disease classifier."""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

from backend.features import encode_features

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / 'frontend'))

# Load trained artifacts once at startup. Run backend/models.py first.
baseline_model = joblib.load(ROOT / 'baseline_model.pkl')
advanced_model = load_model(ROOT / 'advanced_model.keras')
advanced_bundle = joblib.load(ROOT / 'data' / 'advanced_preprocessing.pkl')


@app.get('/')
def home():
    return render_template('homes.html')


@app.post('/predict')
def predict():
    try:
        raw = {
            'Age': int(request.form['Age']),
            'Sex': request.form['Sex'],
            'ChestPainType': request.form['ChestPainType'],
            'RestingBP': float(request.form['RestingBP']),
            'Cholesterol': float(request.form['Cholesterol']),
            'FastingBS': int(request.form['FastingBS']),
            'RestingECG': request.form['RestingECG'],
            'MaxHR': int(request.form['MaxHR']),
            'ExerciseAngina': request.form['ExerciseAngina'],
            'Oldpeak': float(request.form['Oldpeak']),
            'ST_Slope': request.form['ST_Slope'],
        }
        if not (1 <= raw['Age'] <= 120 and 0 <= raw['RestingBP'] <= 300
                and 0 <= raw['Cholesterol'] <= 1000 and raw['FastingBS'] in (0, 1)
                and 0 <= raw['MaxHR'] <= 250 and -10 <= raw['Oldpeak'] <= 20):
            raise ValueError('One or more numeric values are outside the accepted range.')

        X = encode_features(pd.DataFrame([raw]))
        model_choice = request.form.get('model', 'advanced')
        if model_choice == 'baseline':
            # Baseline pipeline includes its own fitted StandardScaler.
            probability = float(baseline_model.predict_proba(X)[:, 1][0])
            threshold = 0.5
            model_name = 'Logistic regression'
        elif model_choice == 'advanced':
            # NN scaler was fitted only on training observations.
            X = X.loc[:, advanced_bundle['feature_names']]
            X_scaled = advanced_bundle['scaler'].transform(X)
            probability = float(advanced_model.predict(X_scaled, verbose=0).ravel()[0])
            threshold = float(advanced_bundle['threshold'])
            model_name = 'Neural network'
        else:
            raise ValueError('Unrecognized model choice.')

        positive = probability >= threshold
        result = {
            'prediction': 'Model predicts heart disease' if positive else 'Model predicts no heart disease',
            'probability': f'{probability * 100:.1f}%',
            'risk': 'Positive classification' if positive else 'Negative classification',
            'model': model_name,
            'threshold': f'{threshold:.3f}',
            'positive': positive,
        }
        return render_template('display.html', result=result)
    except (KeyError, TypeError, ValueError) as exc:
        return render_template('homes.html', error=f'Invalid input: {exc}'), 400


if __name__ == '__main__':
    app.run(debug=False)
