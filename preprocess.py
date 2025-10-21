import gzip
import pandas as pd
import re
from sklearn.model_selection import train_test_split

with gzip.open("/Users/curious_techie/Desktop/sentimental-analysis/data/Video_Games.jsonl.gz", "rt", encoding="utf-8") as f:
    df = pd.read_json(f, lines=True)

# Basic cleaning
df['text'] = df['text'].astype(str).map(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))

# Sentiment label mapping
df['sentiment'] = df['rating'].map(lambda x: "positive" if x > 3 else ("negative" if x < 3 else "neutral"))

# Drop missing
df = df.dropna(subset=['text', 'sentiment'])


# Split
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])

# Store the train and test data in csv file
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
