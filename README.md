SvaraAI — Reply Classifier

This project is a simple but complete pipeline for classifying short text replies. It starts with a traditional ML baseline (TF-IDF + Logistic Regression) and then explores fine-tuning a transformer model (DistilBERT). I also wrapped everything into a FastAPI service, so it can be tested like a real application.

Project Structure

prepare_and_train.py — trains a baseline model using TF-IDF and Logistic Regression.

finetune_transformer.py — script to fine-tune DistilBERT with Hugging Face.

app.py — FastAPI server that exposes a /predict endpoint.

models/ — trained models are stored here.

data/replies.csv — dataset file (with two columns: text, label).

answers.md — short reasoning answers.

requirements.txt — Python dependencies.

Dockerfile — containerization setup (optional).

How to Run
1. Setup Environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt

2. Prepare Dataset

Place the assignment CSV into:

data/replies.csv

3. Train Baseline Model
python prepare_and_train.py


This creates:

models/tfidf_vectorizer.joblib

models/logreg_baseline.joblib

4. (Optional) Fine-Tune DistilBERT

If you have GPU access (local or Colab):

python finetune_transformer.py


This saves a fine-tuned model in:

models/distilbert_finetuned/

5. Run API
uvicorn app:app --reload


Send a POST request to:

http://127.0.0.1:8000/predict


Example:

{ "text": "Looking forward to the demo!" }

Notes & Tips

Keep epochs and batch sizes small if the dataset is tiny.

For production, model versioning and proper serving (TorchServe, Triton, autoscaling containers) should be considered.

See notes_on_adaptation.txt for ideas on how to pitch this project in a recruiter-facing way.

Docker Setup (Optional)

Build image:

docker build -t reply-classifier .


Run container:

docker run -p 8000:8000 reply-classifier
