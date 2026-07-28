import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def preprocess_reviews(raw_data):
    if not raw_data:
        return pd.DataFrame()

    df = pd.DataFrame(raw_data).copy()

    if "content" not in df.columns:
        return pd.DataFrame()

    df["review"] = df["content"].fillna("").astype(str)
    df["clean_review"] = df["review"].apply(clean_text)

    return df

def preprocess_categorized(categorized):
    """Preprocess tiap kategori, return dict of DataFrames"""
    return {
        category: preprocess_reviews(reviews_list)
        for category, reviews_list in categorized.items()
    }