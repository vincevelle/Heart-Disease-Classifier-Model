from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.features import FEATURES, encode_features

DATA_DIR = ROOT / 'data'
RAW_PATH = ROOT / 'heart_disease.csv'
OUTPUT_PATH = DATA_DIR / 'heart_disease_preprocessed.csv'
PLOT_PATH = DATA_DIR / 'correlation_matrix.png'


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH)
    df = df[FEATURES].drop_duplicates().dropna().copy()
    df = encode_features(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved {OUTPUT_PATH} ({len(df)} rows)')

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()
    print(f'Saved {PLOT_PATH}')


if __name__ == '__main__':
    main()
