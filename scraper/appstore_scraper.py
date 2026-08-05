import time

import requests
from predictor.predictor import predict_batch
from preprocessing.sentiment_cleaning import is_indonesian_text

REVIEW_PAGE_SIZE = 50
MAX_REVIEW_PAGES = 10  # feed RSS Apple mentok di ~10 halaman (500 ulasan)
MAX_EMPTY_RETRIES = 2  # feed Apple kadang kosong sesaat walau datanya ada
RETRY_DELAY_SECONDS = 1.5
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _fetch_review_page(app_id, country, page):
    # Catatan: parameter "sortby=mostrecent" sering membuat Apple mengembalikan
    # feed kosong (tanpa "entry") walau aplikasinya punya ratusan ulasan —
    # urutan default (tanpa sortby) jauh lebih konsisten mengembalikan data.
    url = (
        f"https://itunes.apple.com/{country}/rss/customerreviews/"
        f"id={app_id}/page={page}/json"
    )
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _fetch_page_reviews(app_id, country, page):
    """Ambil satu halaman ulasan, dengan retry karena feed Apple kadang
    kosong secara transien walau aplikasinya punya banyak ulasan."""
    for attempt in range(MAX_EMPTY_RETRIES + 1):
        data = _fetch_review_page(app_id, country, page)
        entries = (data.get("feed") or {}).get("entry") or []
        if isinstance(entries, dict):
            entries = [entries]

        page_reviews = [e for e in entries if (e.get("im:rating") or {}).get("label") is not None]
        if page_reviews or attempt == MAX_EMPTY_RETRIES:
            return page_reviews

        time.sleep(RETRY_DELAY_SECONDS)

    return []


def scrape_reviews(app_id, country="id", count=500):
    reviews = []
    try:
        for page in range(1, MAX_REVIEW_PAGES + 1):
            if len(reviews) >= count:
                break

            page_reviews = _fetch_page_reviews(app_id, country, page)
            if not page_reviews:
                break

            for entry in page_reviews:
                content = ((entry.get("content") or {}).get("label") or "").strip()
                if content and not is_indonesian_text(content):
                    continue  # lewati ulasan yang jelas bukan berbahasa Indonesia

                rating = (entry.get("im:rating") or {}).get("label")
                score = int(rating) if str(rating).isdigit() else None
                reviews.append({
                    "reviewId": (entry.get("id") or {}).get("label", ""),
                    "userName": ((entry.get("author") or {}).get("name") or {}).get("label", ""),
                    "content": content,
                    "title": (entry.get("title") or {}).get("label", ""),
                    "score": score,
                    "at": (entry.get("updated") or {}).get("label", ""),
                    "appVersion": (entry.get("im:version") or {}).get("label", ""),
                })

        return reviews[:count]
    except Exception as e:
        print("SCRAPER ERROR:", e)
        return []


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


def search_apps(query, country="id", limit=8):
    url = "https://itunes.apple.com/search"
    params = {
        "term": query,
        "country": country,
        "media": "software",
        "limit": limit,
    }
    resp = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("results") or []:
        app_id = str(item.get("trackId") or "").strip()
        if not app_id:
            continue
        results.append({
            "title": item.get("trackName", ""),
            "developer": item.get("artistName", ""),
            "appId": app_id,
            "icon": item.get("artworkUrl100", ""),
            "url": item.get("trackViewUrl", ""),
        })

    return results
