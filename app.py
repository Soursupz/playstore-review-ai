from flask import Flask, render_template, request, session
from scraper.playstore_scraper import scrape_reviews
from preprocessing.cleaning import preprocess_reviews
from retrieval.search import search_reviews
from llm.llm_module import generate_answer
from analysis.sentiment import simple_sentiment

from urllib.parse import urlparse, parse_qs

app = Flask(__name__)
app.secret_key = "supersecretkey123"


@app.route("/", methods=["GET", "POST"])
def index():

    # 🔥 Inisialisasi chat history
    if "chat_history" not in session:
        session["chat_history"] = []

    error_message = None
    stats = None

    if request.method == "POST":
        try:
            link = request.form.get("link")
            query = request.form.get("query")

            if not link or not query:
                error_message = "Link dan pertanyaan harus diisi."
                return render_template(
                    "index.html",
                    chat_history=session["chat_history"],
                    error=error_message
                )

            # 🔥 Parse link Play Store
            if "http" in link:
                parsed_url = urlparse(link)
                query_params = parse_qs(parsed_url.query)
                package_name = query_params.get("id", [""])[0]
            else:
                package_name = link.strip()

            if not package_name:
                error_message = "Format link tidak valid."
                return render_template(
                    "index.html",
                    chat_history=session["chat_history"],
                    error=error_message
                )

            print("📦 PACKAGE NAME:", package_name)

            # 🔎 SCRAPING
            raw_data = scrape_reviews(package_name)

            if not raw_data:
                error_message = "Review tidak ditemukan."
                return render_template(
                    "index.html",
                    chat_history=session["chat_history"],
                    error=error_message
                )

            print("📝 TOTAL REVIEW:", len(raw_data))

            # 🧹 PREPROCESS
            df = preprocess_reviews(raw_data)

            # 🔍 RETRIEVAL
            relevant_reviews = search_reviews(query, df)

            print("🔎 RELEVANT REVIEWS:", relevant_reviews)

            # 📊 Statistik
            stats = f"{len(raw_data)} review dianalisis"

            # 🤖 LLM
            answer, token_used = generate_answer(query, relevant_reviews)

            # 📈 Sentiment
            sentiment = simple_sentiment(relevant_reviews)

            # 💾 Simpan ke chat history
            session["chat_history"].append({
                "query": query,
                "answer": answer,
                "sentiment": sentiment,
                "token": token_used,
                "stats": stats
            })

            session.modified = True

        except Exception as e:
            print("❌ APP ERROR:", e)
            error_message = "Terjadi kesalahan sistem."

    return render_template(
        "index.html",
        chat_history=session["chat_history"],
        error=error_message
    )


@app.route("/reset")
def reset_chat():
    session.clear()
    return render_template("index.html", chat_history=[])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
