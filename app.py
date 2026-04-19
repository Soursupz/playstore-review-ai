from flask import Flask, render_template, request, session, jsonify
from scraper.playstore_scraper import scrape_reviews, categorize_by_rating, categorize_by_model
from preprocessing.cleaning import preprocess_categorized
from retrieval.search import search_categorized
from llm.llm_module import generate_answer, handle_scraping_error
from analysis.sentiment import sentiment_stats
from predictor.predictor import predict_sentiment, predict_batch
import os
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)
app.secret_key = "supersecretkey123"

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        action = request.form.get("action")

        if action == "change_link":
            session.pop("package_name", None)
            session.modified = True
            return render_template(
                "index.html",
                chat_history=session["chat_history"],
                package_name=None
            )

        try:
            query = request.form.get("query")

            if "package_name" not in session:
                link = request.form.get("link")

                if not link or not query:
                    return render_template(
                        "index.html",
                        chat_history=session["chat_history"],
                        package_name=None,
                        error="Link dan pertanyaan harus diisi."
                    )

                if "http" in link:
                    parsed_url   = urlparse(link)
                    query_params = parse_qs(parsed_url.query)
                    package_name = query_params.get("id", [""])[0]
                else:
                    package_name = link.strip()

                    # ✅ Validasi package name
                if not package_name or len(package_name) < 3 or ' ' in package_name:
                 return render_template(
                    "index.html",
                    chat_history=session["chat_history"],
                    package_name=None,
                    error="Link atau package name tidak valid. Pastikan kamu memasukkan link Play Store yang benar (contoh: https://play.google.com/store/apps/details?id=com.shopee.id) atau package name langsung (contoh: com.shopee.id)"
                     )

                session["package_name"] = package_name

            else:
                package_name = session["package_name"]

            # Scraping
            raw_data = scrape_reviews(package_name)

            if not raw_data:
            # Reset package name biar user bisa input ulang
                session.pop("package_name", None)
                session.modified = True

                error_answer = (
                "Link atau package name yang kamu masukkan tidak valid atau "
                "aplikasi tidak ditemukan di Play Store Indonesia. "
                "Silakan coba lagi dengan:\n"
                "1. Link Play Store yang benar, contoh:\n"
                "   https://play.google.com/store/apps/details?id=com.shopee.id\n"
                "2. Atau package name langsung, contoh:\n"
                "   com.shopee.id / com.gojek.app / com.tokopedia.tkpd"
                )
                
                session["chat_history"].append({
                "query":     query,
                "answer":    error_answer,
                "sentiment": None,
                "token":     0,
                "stats":     "Link tidak valid"
                 })
                
                session.modified = True
                return render_template(
                 "index.html",
                chat_history=session["chat_history"],
                 package_name=None
                )

            # Kategorisasi pakai IndoBERT
            categorized     = categorize_by_model(raw_data)
            categorized_dfs = preprocess_categorized(categorized)
            categorized_results = search_categorized(query, categorized_dfs)

            all_relevant = []
            for reviews_list in categorized_results.values():
                all_relevant.extend(reviews_list)

            sentiment      = sentiment_stats(categorized)
            safe_sentiment = sentiment if isinstance(sentiment, dict) else None
            stats          = f"{len(raw_data)} review dianalisis"

            # Generate jawaban dengan semua fitur baru
            try:
                is_first    = len(session["chat_history"]) == 0
                answer, token_used = generate_answer(
                    query,
                    all_relevant,
                    categorized_results,
                    sentiment=safe_sentiment,
                    chat_history=session["chat_history"],
                    is_first_message=is_first
                )
            except Exception as e:
                print("LLM ERROR:", e)
                answer     = "Error generate jawaban"
                token_used = 0

            session["chat_history"].append({
                "query":     query,
                "answer":    str(answer),
                "sentiment": safe_sentiment,
                "token":     token_used,
                "stats":     stats
            })
            session.modified = True

        except Exception as e:
            print("ERROR:", e)
            error_answer, _ = handle_scraping_error()
            session["chat_history"].append({
                "query":     request.form.get("query", ""),
                "answer":    error_answer,
                "sentiment": None,
                "token":     0,
                "stats":     "Scraping gagal"
            })
            session.modified = True
            return render_template(
                "index.html",
                chat_history=session["chat_history"],
                package_name=session.get("package_name")
            )

    return render_template(
        "index.html",
        chat_history=session["chat_history"],
        package_name=session.get("package_name")
    )


@app.route("/reset")
def reset_chat():
    session.clear()
    return render_template("index.html", chat_history=[], package_name=None)


# ============================================================
# Endpoint: Prediksi Single Review
# ============================================================
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Teks tidak boleh kosong"}), 400

        result = predict_sentiment(text)
        return jsonify({
            "status": "success",
            "text":   text,
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Endpoint: Prediksi Batch Review
# ============================================================
@app.route("/api/predict/batch", methods=["POST"])
def predict_batch_reviews():
    try:
        data  = request.get_json()
        texts = data.get("texts", [])

        if not texts:
            return jsonify({"error": "Texts tidak boleh kosong"}), 400

        results = predict_batch(texts)
        total   = len(results)
        positif = sum(1 for r in results if r["label"] == "positif")
        negatif = total - positif

        return jsonify({
            "status": "success",
            "total":  total,
            "statistik": {
                "positif": {
                    "count":      positif,
                    "percentage": round(positif / total * 100, 1)
                },
                "negatif": {
                    "count":      negatif,
                    "percentage": round(negatif / total * 100, 1)
                }
            },
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)