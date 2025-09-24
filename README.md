# SvaraAI — Reply Classifier (Sample / Learning Resource)

> IMPORTANT: This repository is a **sample learning resource**. Use it to learn and adapt; do not submit it verbatim as your own assignment.

## Contents
- `prepare_and_train.py` — baseline TF-IDF + LogisticRegression training.
- `finetune_transformer.py` — fine-tune DistilBERT using Hugging Face.
- `app.py` — FastAPI service with `/predict`.
- `models/` — where trained models will be saved.
- `data/replies.csv` — put the provided CSV here (`text`, `label` columns).
- `answers.md` — short reasoning answers.
- `requirements.txt` — packages.
- `Dockerfile` — optional containerization.

## Quick start (local / Colab)
1. Create virtual environment and install:
   ```bash
   python -m venv venv
   source venv/bin/activate       # on Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. Put the assignment CSV at `data/replies.csv`.

3. Train baseline:
   ```bash
   python prepare_and_train.py
   ```
   This produces `models/tfidf_vectorizer.joblib` and `models/logreg_baseline.joblib`.

4. (Optional but recommended) Fine-tune transformer (requires GPU):
   ```bash
   python finetune_transformer.py
   ```
   This produces `models/distilbert_finetuned/`.

5. Run API:
   ```bash
   uvicorn app:app --reload
   ```
   POST to `http://127.0.0.1:8000/predict` with JSON:
   ```json
   { "text": "Looking forward to the demo!" }
   ```

## Notes
- Use small epochs and small batch sizes when dataset is small.
- For real deployment, use model versioning and a model-serving framework (TorchServe, Triton, or an autoscaled container).
- See `notes_on_adaptation.txt` for suggestions on writing a recruiter-facing project description.

## Docker (optional)
Build:
```bash
docker build -t reply-classifier:sample .
docker run -p 8000:8000 reply-classifier:sample
```
