"""Shared feature schema and encoding for training and Flask inference."""
import pandas as pd

FEATURES = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope', 'HeartDisease',
]
CATEGORIES = {
    'Sex': ['F', 'M'],
    'ChestPainType': ['ASY', 'ATA', 'NAP', 'TA'],
    'RestingECG': ['LVH', 'Normal', 'ST'],
    'ExerciseAngina': ['N', 'Y'],
    'ST_Slope': ['Down', 'Flat', 'Up'],
}
NUMERIC_FEATURES = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']


def encode_features(raw: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode known categories and add deterministic interaction features."""
    df = raw.copy()
    # Treat invalid zero-valued measurements as missing at training and inference.
    df[['RestingBP', 'Cholesterol']] = df[['RestingBP', 'Cholesterol']].replace(0, float('nan'))
    for name, categories in CATEGORIES.items():
        unknown = set(df[name].dropna().unique()) - set(categories)
        if unknown:
            raise ValueError(f'Unexpected {name} categories: {sorted(unknown)}')
        df[name] = pd.Categorical(df[name], categories=categories)
    df = pd.get_dummies(df, columns=list(CATEGORIES), drop_first=True, dtype=int)
    df['Oldpeak_x_ExerciseAngina'] = df['Oldpeak'] * df['ExerciseAngina_Y']
    df['MaxHR_x_Age'] = df['MaxHR'] * df['Age']
    df['ST_Slope_Flat_x_Oldpeak'] = df['ST_Slope_Flat'] * df['Oldpeak']
    return df
