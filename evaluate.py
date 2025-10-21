import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("outputs/best_model")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model.eval()  # set model to evaluation mode

test_df = pd.read_csv("data/test.csv")
label_map = {'negative':0, 'neutral':1, 'positive':2}
test_df['label'] = test_df['sentiment'].map(label_map)


texts = test_df['text'].fillna("").astype(str).tolist()

# Batch inference
batch_size = 512
all_preds = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i: i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

test_df['pred'] = all_preds
print(classification_report(test_df['label'], test_df['pred']))