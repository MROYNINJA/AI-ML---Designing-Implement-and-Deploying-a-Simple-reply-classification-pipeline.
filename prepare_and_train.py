# prepare_and_train.py
"""
Sample script: load dataset, preprocess, train baseline (TF-IDF + LogisticRegression),
evaluate (accuracy, macro F1), save baseline model and TF-IDF vectorizer.
This is a learning/sample resource — adapt it and understand it before using.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import re
import string

# CONFIG
DATA_PATH = "data/replies.csv"   # update path
OUTPUT_DIR = "models"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_data(path):
    df = pd.read_csv(path)
    # Expected columns: text, label
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    return df[['text','label']].copy()

def clean_text(s):
    if pd.isna(s):
        return ""
    s = s.lower()
    s = re.sub(r'\\s+', ' ', s)  # normalize whitespace
    s = re.sub(r'https?://\\S+|www\\.\\S+', ' ', s)  # remove urls
    s = re.sub(r'[\\r\\n]+', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.strip()
    return s

def prepare(df):
    df['text_clean'] = df['text'].astype(str).map(clean_text)
    # drop empty texts
    df = df[~df['text_clean'].str.strip().eq('')].reset_index(drop=True)
    # map labels to canonical form
    df['label'] = df['label'].str.lower().str.strip()
    return df

def train_baseline(X_train, y_train):
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
    Xtr = vect.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
    clf.fit(Xtr, y_train)
    return vect, clf

def evaluate(vect, clf, X, y_true):
    Xtf = vect.transform(X)
    y_pred = clf.predict(Xtf)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred)
    return acc, f1, report

def main():
    df = read_data(DATA_PATH)
    df = prepare(df)

    train, test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=RANDOM_STATE)
    train, val = train_test_split(train, test_size=0.125, stratify=train['label'], random_state=RANDOM_STATE)  # 70/15/15

    print("Counts:", train['label'].value_counts().to_dict())
    vect, clf = train_baseline(train['text_clean'].tolist(), train['label'].tolist())

    acc_val, f1_val, rep_val = evaluate(vect, clf, val['text_clean'].tolist(), val['label'].tolist())
    acc_test, f1_test, rep_test = evaluate(vect, clf, test['text_clean'].tolist(), test['label'].tolist())

    print("Validation Acc:", acc_val, "F1_macro:", f1_val)
    print(rep_val)
    print("Test Acc:", acc_test, "F1_macro:", f1_test)
    print(rep_test)

    # Save artifacts
    joblib.dump(vect, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(OUTPUT_DIR, "logreg_baseline.joblib"))

    # Save metrics
    metrics = {
        "val": {"accuracy": float(acc_val), "f1_macro": float(f1_val)},
        "test": {"accuracy": float(acc_test), "f1_macro": float(f1_test)}
    }
    with open(os.path.join(OUTPUT_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
