from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger('fusion_train')


def find_cols(df, prefix):
    return [c for c in df.columns if str(c).startswith(prefix)]


def main():
    parser = argparse.ArgumentParser(description="Train fusion meta-model from per-modality probabilities CSV")
    parser.add_argument('--data-csv', '-i', required=True, help='CSV with per-modality probabilities and label')
    parser.add_argument('--out-dir', '-o', default='models/fusion_model', help='Output directory for fusion model')
    parser.add_argument('--model', choices=['logreg', 'mlp'], default='logreg', help='Meta-model type')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data for testing')
    parser.add_argument('--label-col', default='label', help='Name of the label column')
    parser.add_argument('--verbose', action='store_true', help='Print confusion matrix')
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    fer_cols = find_cols(df, 'fer_')
    speech_cols = find_cols(df, 'speech_')
    text_cols = find_cols(df, 'text_')

    if len(fer_cols) == 0:
        logger.error('No FER columns detected (expected columns starting with "fer_")')
        return

    if args.label_col not in df.columns:
        logger.error(f'Label column "{args.label_col}" not found in CSV')
        return

    feature_cols = fer_cols + speech_cols + text_cols
    X = df[feature_cols].fillna(0.0).astype(np.float32).values
    y = df[args.label_col].astype(str).values

    # Stratified split (safe fallback)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Using non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )

    # Model selection
    if args.model == 'logreg':
        clf = LogisticRegression(multi_class='multinomial', max_iter=2000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)

    logger.info(f'Training {args.model} meta-model on {X_train.shape[0]} samples...')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    logger.info('Test classification report:\n%s', classification_report(y_test, y_pred))

    if args.verbose:
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        logger.info('Confusion matrix:\n%s', cm)

    # Save model payload
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': clf,
        'meta': {
            'feature_columns': feature_cols,
            'n_fer': len(fer_cols),
            'n_speech': len(speech_cols),
            'n_text': len(text_cols)
        }
    }

    out_path = out_dir / 'fusion_model.pkl'
    joblib.dump(payload, out_path)
    logger.info(f'Fusion model saved to: {out_path}')


if __name__ == '__main__':
    main()
