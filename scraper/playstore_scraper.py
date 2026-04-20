from google_play_scraper import reviews, Sort
from predictor.predictor import predict_sentiment, predict_batch

def scrape_reviews(package_name, count=300):
    try:
        result, _ = reviews(
            package_name,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=count
        )
        return result
    except Exception as e:
        print("SCRAPER ERROR:", e)
        return []

def categorize_by_rating(raw_data):
    bad, neutral, good = [], [], []

    for review in raw_data:
        score = review.get("score", 3)
        if score <= 2:
            bad.append(review)
        elif score == 3:
            neutral.append(review)
        else:
            good.append(review)

    return {"bad": bad, "neutral": neutral, "good": good}


def categorize_by_model(raw_data):
    if not raw_data:
        return {"bad": [], "good": []}

    # Pisahkan review kosong dulu
    valid   = [(i, r) for i, r in enumerate(raw_data) if r.get("content", "").strip()]
    invalid = [(i, r) for i, r in enumerate(raw_data) if not r.get("content", "").strip()]

    print(f"📦 Total review: {len(raw_data)} | Valid: {len(valid)} | Kosong: {len(invalid)}")

    # Batch predict semua sekaligus — jauh lebih cepat
    texts   = [r.get("content", "").strip() for _, r in valid]
    batch_results = predict_batch(texts)

    bad, good = [], []

    # Masukkan review kosong langsung ke bad
    for _, review in invalid:
        review["sentiment_label"]      = "negatif"
        review["sentiment_confidence"] = 0.0
        review["sentiment_scores"]     = {"negatif": 100.0, "positif": 0.0}
        bad.append(review)

    # Proses hasil batch
    for (_, review), result in zip(valid, batch_results):
        review["sentiment_label"]      = result["label"]
        review["sentiment_confidence"] = result["confidence"]
        review["sentiment_scores"]     = result["scores"]

        if result["label"] == "positif":
            good.append(review)
        else:
            bad.append(review)

    print(f"✅ Kategorisasi IndoBERT selesai:")
    print(f"   Positif : {len(good)}")
    print(f"   Negatif : {len(bad)}")

    return {"bad": bad, "good": good}