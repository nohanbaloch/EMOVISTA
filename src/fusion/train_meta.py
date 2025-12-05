"""Train a small fusion meta-model from precomputed per-modality probability features.

Expected input: a CSV with columns naming per-modality probability features and
a ground-truth label. Column naming convention (auto-detected):

- FER probabilities: columns starting with `fer_` (e.g. `fer_0` .. `fer_6`)
- Speech probabilities: columns starting with `speech_` (e.g. `speech_0` ..)
- Text probabilities: columns starting with `text_` (e.g. `text_0` ..)
- Label column: default name `label` (can be overridden with --label-col)

The script trains a multinomial LogisticRegression (or MLP) on the
concatenated probability vectors and saves a joblib payload to
`models/fusion_model/fusion_model.pkl` with keys `model` and `meta`.
"""
from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


def find_cols(df, prefix):
    return [c for c in df.columns if str(c).startswith(prefix)]


def main():
    parser = argparse.ArgumentParser(description="Train fusion meta-model from CSV of per-modality probabilities")
    parser.add_argument('--data-csv', '-i', required=True, help='Path to CSV containing per-modality probabilities and labels')
    parser.add_argument('--out-dir', '-o', default='models/fusion_model', help='Output directory for trained fusion model')
    parser.add_argument('--model', choices=['logreg', 'mlp'], default='logreg', help='Type of meta-model to train')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data to reserve for testing')
    parser.add_argument('--label-col', default='label', help='Name of the ground-truth label column')
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    fer_cols = find_cols(df, 'fer_')
    speech_cols = find_cols(df, 'speech_')
    text_cols = find_cols(df, 'text_')

    if len(fer_cols) == 0:
        raise SystemExit('No FER columns detected (expected columns starting with "fer_")')

    if args.label_col not in df.columns:
        raise SystemExit(f'Label column "{args.label_col}" not found in CSV')

    feature_cols = fer_cols + speech_cols + text_cols
    X = df[feature_cols].fillna(0.0).values.astype(float)
    y = df[args.label_col].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)

    if args.model == 'logreg':
        clf = LogisticRegression(multi_class='multinomial', max_iter=2000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)

    print('Training meta-model on', X_train.shape[0], 'samples...')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('\nTest classification report:')
    print(classification_report(y_test, y_pred))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        'model': clf,
        'meta': {
            'feature_columns': feature_cols,
            'n_fer': len(fer_cols),
            'n_speech': len(speech_cols),
            'n_text': len(text_cols),
        }
    }

    out_path = out_dir / 'fusion_model.pkl'
    joblib.dump(payload, out_path)
    print(f'Fusion model saved to: {out_path}')


if __name__ == '__main__':
    main()
