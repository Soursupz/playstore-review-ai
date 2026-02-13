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

    df = pd.DataFrame(raw_data)

    if "content" not in df.columns:
        return pd.DataFrame()

    df = df[["content"]]
    df.rename(columns={"content": "review"}, inplace=True)
    df["clean_review"] = df["review"].apply(clean_text)

    return df
