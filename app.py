# app.py
"""
FastAPI service with a /predict endpoint.
It will try to load a HF transformer model; if not available, it loads TF-IDF + LogisticRegression baseline.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import json
import numpy as np
import uvicorn

app = FastAPI(title="Reply Classifier (sample)")

MODEL_DIR = "models"
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
LOGREG_PATH = os.path.join(MODEL_DIR, "logreg_baseline.joblib")
DISTILBERT_DIR = os.path.join(MODEL_DIR, "distilbert_finetuned")

# Input schema
class PredictRequest(BaseModel):
    text: str

# Attempt to load transformer
transformer_loaded = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    if os.path.isdir(DISTILBERT_DIR):
        tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
        with open(os.path.join(DISTILBERT_DIR, "label_list.json"), "r") as f:
            label_list = json.load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        transformer_loaded = True
except Exception as e:
    transformer_loaded = False

# If transformer not loaded, load baseline
baseline_loaded = False
if not transformer_loaded:
    try:
        vect = joblib.load(TFIDF_PATH)
        clf = joblib.load(LOGREG_PATH)
        # need label list -> infer from clf.classes_ if available
        label_list = list(clf.classes_)
        baseline_loaded = True
    except Exception as e:
        baseline_loaded = False

if not transformer_loaded and not baseline_loaded:
    raise RuntimeError("No model available. Please train models and place them under models/")

@app.get("/")
def root():
    return {"info": "Reply classifier sample API. POST /predict with JSON {text:...}"}

def transform_and_predict_text(text):
    # simple cleaning (consistent with training)
    text_proc = text.lower().strip()
    X = vect.transform([text_proc])
    probs = clf.predict_proba(X)[0]
    idx = np.argmax(probs)
    return {"label": label_list[idx], "confidence": float(probs[idx])}

def transformer_predict(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k:v.to(device) for k,v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()[0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    idx = int(np.argmax(probs))
    return {"label": label_list[idx], "confidence": float(probs[idx])}

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text
    if transformer_loaded:
        res = transformer_predict(text)
        res['model'] = 'distilbert'
        return res
    else:
        res = transform_and_predict_text(text)
        res['model'] = 'baseline_logreg'
        return res

# run with: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
