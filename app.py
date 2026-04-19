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
# Menyimpan status scraping tiap user berdasarkan job_id
# Format: { job_id: { "status": "...", "result": {...}, "error": "..." } }
# ============================================================
jobs = {}
jobs_lock = threading.Lock()
 
 
def run_scraping_job(job_id, package_name, query, chat_history):
    """
    Fungsi ini berjalan di background thread.
    Melakukan scraping + kategorisasi + generate jawaban AI.
    Hasilnya disimpan ke dict `jobs`.
    """
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "scraping"
 
        # 1. Scraping
        raw_data = scrape_reviews(package_name)
 
        if not raw_data:
            with jobs_lock:
                jobs[job_id]["status"]  = "error"
                jobs[job_id]["error"]   = (
                    "Link atau package name tidak valid atau aplikasi tidak ditemukan "
                    "di Play Store Indonesia. Silakan coba lagi dengan link yang benar, "
                    "contoh: https://play.google.com/store/apps/details?id=com.shopee.id"
                )
            return
 
        with jobs_lock:
            jobs[job_id]["status"] = "analyzing"
 
        # 2. Kategorisasi IndoBERT (proses terberat)
        categorized      = categorize_by_model(raw_data)
        categorized_dfs  = preprocess_categorized(categorized)
        categorized_results = search_categorized(query, categorized_dfs)
 
        all_relevant = []
        for reviews_list in categorized_results.values():
            all_relevant.extend(reviews_list)
 
        sentiment      = sentiment_stats(categorized)
        safe_sentiment = sentiment if isinstance(sentiment, dict) else None
        stats          = f"{len(raw_data)} review dianalisis"
 
        with jobs_lock:
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
        with jobs_lock:
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
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"]  = str(e)
 
 
# ============================================================
# ROUTE: Halaman utama (GET)
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
# ROUTE: Start scraping + first question (POST async)
# ============================================================
@app.route("/start", methods=["POST"])
def start():
    """
    Menerima link + query pertama.
    Langsung return job_id, lalu scraping jalan di background.
    """
    if "chat_history" not in session:
        session["chat_history"] = []
 
    data  = request.get_json()
    link  = (data.get("link") or "").strip()
    query = (data.get("query") or "Bagaimana sentimen pengguna secara keseluruhan?").strip()
 
    # Parse package name
    if "http" in link:
        parsed      = urlparse(link)
        params      = parse_qs(parsed.query)
        package_name = params.get("id", [""])[0]
    else:
        package_name = link.strip()
 
    if not package_name or len(package_name) < 3 or ' ' in package_name:
        return jsonify({"error": "Link atau package name tidak valid."}), 400
 
    session["package_name"] = package_name
    session.modified = True
 
    # Buat job baru
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "result": None, "error": None}
 
    # Jalankan di background thread
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
    """
    Frontend polling endpoint ini tiap 3 detik.
    Return status: queued | scraping | analyzing | generating | done | error
    """
    with jobs_lock:
        job = jobs.get(job_id)
 
    if not job:
        return jsonify({"error": "Job tidak ditemukan"}), 404
 
    response = {"status": job["status"]}
 
    if job["status"] == "done":
        result = job["result"]
        response["data"] = result
 
        # Simpan ke session chat history
        session["chat_history"].append(result)
        session.modified = True
 
        # Cleanup job dari memory
        with jobs_lock:
            jobs.pop(job_id, None)
 
    elif job["status"] == "error":
        response["error"] = job["error"]
 
        # Reset package name supaya user bisa input ulang
        session.pop("package_name", None)
        session.modified = True
 
        with jobs_lock:
            jobs.pop(job_id, None)
 
    return jsonify(response)
 
 
# ============================================================
# ROUTE: Tanya lanjutan (tanpa scraping ulang)
# ============================================================
@app.route("/ask", methods=["POST"])
def ask():
    """
    Untuk pertanyaan ke-2 dst — package_name sudah di session,
    jadi langsung scraping + generate tanpa perlu job polling
    (atau bisa juga pakai polling, tergantung kebutuhan).
    Kita pakai async juga supaya konsisten.
    """
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
    with jobs_lock:
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
# ROUTE: Reset seluruh sesi
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