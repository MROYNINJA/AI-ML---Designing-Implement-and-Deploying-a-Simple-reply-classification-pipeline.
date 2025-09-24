# finetune_transformer.py
"""
Sample script: fine-tune distilbert-base-uncased on the labeled replies.
Requires: transformers, datasets, torch
"""
import os
import pandas as pd
import numpy as np
from datasets import Dataset, ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import json

DATA_PATH = "data/replies.csv"
OUTPUT_DIR = "models/distilbert_finetuned"
MODEL_NAME = "distilbert-base-uncased"
RANDOM_SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)[['text','label']].dropna()
    df['label'] = df['label'].str.lower().str.strip()
    # keep only known labels
    label_list = sorted(df['label'].unique().tolist())
    label_to_id = {l:i for i,l in enumerate(label_list)}
    df['label_id'] = df['label'].map(label_to_id)
    return df, label_list, label_to_id

def preprocess(tokenizer, texts, max_length=128):
    return tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)

def compute_metrics(pred):
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)['accuracy']
    f1_macro = metric_f1.compute(predictions=preds, references=labels, average='macro')['f1']
    return {"accuracy": acc, "f1_macro": f1_macro}

def main():
    df, label_list, label_to_id = load_and_prepare()
    # train/val/test split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=RANDOM_SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.1765, stratify=train_df['label'], random_state=RANDOM_SEED)  # ~70/15/15

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = Dataset.from_pandas(train_df[['text','label_id']].rename(columns={'label_id':'label'}))
    val_ds = Dataset.from_pandas(val_df[['text','label_id']].rename(columns={'label_id':'label'}))
    test_ds = Dataset.from_pandas(test_df[['text','label_id']].rename(columns={'label_id':'label'}))

    def tok_batch(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
    train_ds = train_ds.map(tok_batch, batched=True)
    val_ds = val_ds.map(tok_batch, batched=True)
    test_ds = test_ds.map(tok_batch, batched=True)
    train_ds = train_ds.remove_columns(['text'])
    val_ds = val_ds.remove_columns(['text'])
    test_ds = test_ds.remove_columns(['text'])
    train_ds.set_format(type='torch')
    val_ds.set_format(type='torch')
    test_ds.set_format(type='torch')

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        seed=RANDOM_SEED,
    )

    def compute_metrics_hf(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        from sklearn.metrics import f1_score, accuracy_score
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average='macro'))
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_hf,
    )

    trainer.train()
    metrics = trainer.evaluate(test_ds)
    print("Test metrics:", metrics)

    # Save model + tokenizer + label mapping
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "label_list.json"), "w") as f:
        json.dump(label_list, f)

if __name__ == "__main__":
    main()
