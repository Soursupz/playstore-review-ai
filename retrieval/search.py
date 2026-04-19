from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def search_reviews(query, df, top_n=2):
    if df.empty:
        return []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["clean_review"])

    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)

    top_indices = similarity[0].argsort()[-top_n:][::-1]

    return df.iloc[top_indices]["review"].tolist()

def search_categorized(query, categorized_dfs, top_n=2):
    """
    Search di tiap kategori, return dict berisi
    review relevan per kategori
    """
    results = {}
    for category, df in categorized_dfs.items():
        results[category] = search_reviews(query, df, top_n)
    return results