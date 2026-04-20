from flask import Flask, render_template, request, session, jsonify
from scraper.playstore_scraper import scrape_reviews, categorize_by_model
from preprocessing.cleaning import preprocess_categorized
from retrieval.search import search_categorized
from llm.llm_module import generate_answer, handle_scraping_error
from analysis.sentiment import sentiment_stats
from predictor.predictor import predict_sentiment, predict_batch
import os
import uuid
import threading
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)
app.secret_key = "supersecretkey123"

# ============================================================
# IN-MEMORY JOB STORE
# Tidak pakai lock — dict operations di Python sudah GIL-safe untuk read/write sederhana
# ============================================================
jobs = {}


def run_scraping_job(job_id, package_name, query, chat_history):
    try:
        jobs[job_id]["status"] = "scraping"

        # 1. Scraping
        raw_data = scrape_reviews(package_name)

        if not raw_data:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"]  = (
                "Link atau package name tidak valid atau aplikasi tidak ditemukan "
                "di Play Store Indonesia. Silakan coba lagi dengan link yang benar, "
                "contoh: https://play.google.com/store/apps/details?id=com.shopee.id"
            )
            return

        jobs[job_id]["status"] = "analyzing"

        # 2. Kategorisasi IndoBERT
        categorized         = categorize_by_model(raw_data)
        categorized_dfs     = preprocess_categorized(categorized)
        categorized_results = search_categorized(query, categorized_dfs)

        all_relevant = []
        for reviews_list in categorized_results.values():
            all_relevant.extend(reviews_list)

        sentiment      = sentiment_stats(categorized)
        safe_sentiment = sentiment if isinstance(sentiment, dict) else None
        stats          = f"✅ {len(raw_data)} data review berhasil diambil"

        jobs[job_id]["status"] = "generating"

        # 3. Generate jawaban AI
        is_first = len(chat_history) == 0
        try:
            answer, token_used = generate_answer(
                query,
                all_relevant,
                categorized_results,
                sentiment=safe_sentiment,
                chat_history=chat_history,
                is_first_message=is_first
            )
        except Exception as e:
            print("LLM ERROR:", e)
            answer     = "Error saat generate jawaban AI."
            token_used = 0

        # 4. Simpan hasil
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "query":     query,
            "answer":    str(answer),
            "sentiment": safe_sentiment,
            "token":     token_used,
            "stats":     stats,
        }

    except Exception as e:
        print("JOB ERROR:", e)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)


# ============================================================
# ROUTE: Halaman utama
# ============================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template(
        "index.html",
        chat_history=session["chat_history"],
        package_name=session.get("package_name")
    )


# ============================================================
# ROUTE: Start scraping
# ============================================================
@app.route("/start", methods=["POST"])
def start():
    if "chat_history" not in session:
        session["chat_history"] = []

    data  = request.get_json()
    link  = (data.get("link") or "").strip()
    query = (data.get("query") or "Hi PSAI").strip()

    # Parse package name
    if "http" in link or link.startswith("www."):
        parsed = urlparse(link if link.startswith("http") else "https://" + link)
        if parsed.netloc not in ("play.google.com", "www.play.google.com"):
            return jsonify({"error": "Link tidak valid. Hanya link aplikasi dari Google Play Store yang diterima. Contoh: https://play.google.com/store/apps/details?id=com.shopee.id"}), 400
        params = parse_qs(parsed.query)
        package_name = params.get("id", [""])[0].split("&")[0].strip()
        if not package_name:
            return jsonify({"error": "Link Play Store tidak mengandung ID aplikasi. Pastikan URL mengandung ?id=nama.paket.aplikasi"}), 400
    else:
        package_name = link.strip()
        if not package_name or ' ' in package_name or '.' not in package_name or len(package_name) < 5:
            return jsonify({"error": "Input tidak valid. Masukkan link Play Store (https://play.google.com/...) atau package name aplikasi (contoh: com.shopee.id)"}), 400

    session["package_name"] = package_name
    session.modified = True

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None, "error": None}

    t = threading.Thread(
        target=run_scraping_job,
        args=(job_id, package_name, query, list(session["chat_history"])),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id, "package_name": package_name})


# ============================================================
# ROUTE: Poll status job
# ============================================================
@app.route("/poll/<job_id>", methods=["GET"])
def poll(job_id):
    job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job tidak ditemukan"}), 404

    response = {"status": job["status"]}

    if job["status"] == "done":
        response["data"] = job["result"]
        session["chat_history"].append(job["result"])
        session.modified = True
        jobs.pop(job_id, None)

    elif job["status"] == "error":
        response["error"] = job["error"]
        session.pop("package_name", None)
        session.modified = True
        jobs.pop(job_id, None)

    return jsonify(response)


# ============================================================
# ROUTE: Tanya lanjutan
# ============================================================
@app.route("/ask", methods=["POST"])
def ask():
    if "chat_history" not in session:
        session["chat_history"] = []

    data  = request.get_json()
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Pertanyaan tidak boleh kosong"}), 400

    package_name = session.get("package_name")
    if not package_name:
        return jsonify({"error": "Tidak ada aplikasi yang aktif. Silakan input link dulu."}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None, "error": None}

    t = threading.Thread(
        target=run_scraping_job,
        args=(job_id, package_name, query, list(session["chat_history"])),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id})


# ============================================================
# ROUTE: Ganti aplikasi
# ============================================================
@app.route("/change", methods=["GET", "POST"])
def change_app():
    session.pop("package_name", None)
    session.modified = True
    return jsonify({"ok": True})


# ============================================================
# ROUTE: Reset sesi
# ============================================================
@app.route("/reset", methods=["GET", "POST"])
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
        return jsonify({"status": "success", "text": text, "result": result})
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
                "positif": {"count": positif, "percentage": round(positif / total * 100, 1)},
                "negatif": {"count": negatif, "percentage": round(negatif / total * 100, 1)}
            },
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)