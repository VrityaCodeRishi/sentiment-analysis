import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

st.set_page_config(page_title="Customer Sentiment Dashboard", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return DistilBertForSequenceClassification.from_pretrained('outputs/best_model'), \
           DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model, tokenizer = load_model()
model.eval()

df = pd.read_csv("data/train.csv")
if "sentiment" not in df.columns:
    st.error("No sentiment column detected.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
sentiment_options = list(df['sentiment'].unique())
selected_sentiment = st.sidebar.multiselect("Sentiment type", sentiment_options, default=sentiment_options)
show_sample = st.sidebar.slider("Reviews Sample Size", 100, len(df), 1000)
filtered_df = df[df['sentiment'].isin(selected_sentiment)].sample(n=min(show_sample, len(df)), random_state=42)

# Dashboard title
st.title("Customer Sentiment Dashboard")
st.markdown("#### Overview of sentiment in customer reviews")

# Sentiment distribution
count_data = filtered_df['sentiment'].value_counts().sort_index()
st.subheader("Sentiment Distribution")
st.bar_chart(count_data)

# Trend over time
if 'reviewTime' in df.columns:
    trend = filtered_df.groupby(['reviewTime', 'sentiment']).size().unstack().fillna(0)
    st.subheader("Sentiment Trend Over Time")
    st.line_chart(trend)

# Show top positive/negative reviews
st.subheader("Sample Reviews")
for label in ['positive', 'negative']:
    st.markdown(f"**{label.title()} reviews:**")
    samples = filtered_df[filtered_df['sentiment'] == label].head(2)
    for review in samples['text']:
        st.write(f"- {review}")

# Real-time prediction
st.subheader("Real-time Sentiment Prediction")
review_text = st.text_area("Enter a review to analyze")
if st.button("Analyze sentiment") and review_text.strip():
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, axis=1).item()
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    st.success(f"Predicted Sentiment: **{mapping[pred]}**")

if st.checkbox("Show Wordclouds (slow)"):
    from wordcloud import WordCloud
    for sentiment in sentiment_options:
        st.markdown(f"**{sentiment.title()} Reviews Wordcloud**")
        text_subset = " ".join(filtered_df[filtered_df['sentiment']==sentiment]['text'].dropna().astype(str))
        if text_subset.strip():
            fig, ax = plt.subplots()
            wc = WordCloud(width=400, height=200, background_color='white').generate(text_subset)
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info(f"No {sentiment} reviews found.")
