from google_play_scraper import reviews, Sort
from predictor.predictor import predict_sentiment

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
    bad, good = [], []

    for review in raw_data:
        text = review.get("content", "").strip()

        if not text:
            bad.append(review)
            continue

        try:
            result = predict_sentiment(text)

            review["sentiment_label"]      = result["label"]
            review["sentiment_confidence"] = result["confidence"]
            review["sentiment_scores"]     = result["scores"]

            if result["label"] == "positif":
                good.append(review)
            else:
                bad.append(review)

        except Exception as e:
            print(f"PREDICT ERROR: {e}")
            bad.append(review)

    print(f"✅ Kategorisasi IndoBERT selesai:")
    print(f"   Positif : {len(good)}")
    print(f"   Negatif : {len(bad)}")

    return {"bad": bad, "good": good}