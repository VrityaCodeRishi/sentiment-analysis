import os
import mlflow
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
from datasets import Dataset

# Load and sample data (use small sample for local testing, scale up on GPU/cloud)
train_df = pd.read_csv("data/train.csv")
label_map = {'negative':0, 'neutral':1, 'positive':2}
train_df['label'] = train_df['sentiment'].map(label_map)

# Sample ~2000 rows for memory-efficient local demo
sample_size = min(len(train_df), 2000)
train_df = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

# Validation split (10% for val)
val_df = train_df.sample(frac=0.1, random_state=42)
train_df = train_df.drop(val_df.index)

# Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_texts = train_df['text'].fillna("").astype(str).tolist()
val_texts = val_df['text'].fillna("").astype(str).tolist()

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# In-memory cleanup
del train_texts, val_texts  # Free RAM

# Datasets
train_dataset = Dataset.from_dict({**train_encodings, 'labels': train_df['label'].tolist()})
val_dataset = Dataset.from_dict({**val_encodings, 'labels': val_df['label'].tolist()})

# Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,  # For testing; increase for real training
    per_device_train_batch_size=4,  # Lower for memory
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    logging_dir='./logs',
    logging_steps=20,
    save_strategy="epoch",
    report_to=["mlflow"],
    disable_tqdm=False,
    load_best_model_at_end=True,
    # metric_for_best_model="eval_accuracy"
)

# MLflow tracking
mlflow.start_run(run_name="distilbert-sentiment-local")
mlflow.log_param("sample_size", sample_size)
mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
mlflow.log_param("epochs", training_args.num_train_epochs)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train & Evaluate
trainer.train()

result_metrics = trainer.evaluate()
for k, v in result_metrics.items():
    mlflow.log_metric(k, v)

model.save_pretrained("outputs/best_model")

mlflow.log_artifact("outputs/best_model")
mlflow.end_run()

del model, trainer
torch.cuda.empty_cache()
