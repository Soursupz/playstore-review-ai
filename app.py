from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request, session, jsonify
from google_play_scraper import search
from scraper.playstore_scraper import scrape_reviews, categorize_by_model
from preprocessing.cleaning import preprocess_categorized
from retrieval.search import search_categorized
from llm.llm_module import generate_answer, handle_scraping_error
from analysis.sentiment import sentiment_stats
from predictor.predictor import predict_sentiment, predict_batch
import os
import uuid
import threading
import time
import json
import hashlib
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)
app.secret_key = "supersecretkey123"

# ============================================================
# IN-MEMORY JOB STORE
# Tidak pakai lock — dict operations di Python sudah GIL-safe untuk read/write sederhana
# ============================================================
jobs = {}
_CACHE_TTL_SECONDS = 1800
_ANALYSIS_CACHE = {}
_SEARCH_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _make_cache_key(*parts):
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(cache, key):
    with _CACHE_LOCK:
        item = cache.get(key)
        if not item:
            return None

        value, expires_at = item
        if time.time() > expires_at:
            cache.pop(key, None)
            return None

        return value


def _cache_set(cache, key, value, ttl=_CACHE_TTL_SECONDS):
    with _CACHE_LOCK:
        cache[key] = (value, time.time() + ttl)


def normalize_package_name(text):
    text = (text or "").strip()
    if not text:
        return ""

    if "play.google.com" in text:
        parsed = urlparse(text)
        params = parse_qs(parsed.query)
        if "id" in params and params["id"]:
            return params["id"][0].strip()

    return text


def get_app_analysis(package_name):
    cache_key = _make_cache_key("analysis", (package_name or "").strip().lower())
    cached = _cache_get(_ANALYSIS_CACHE, cache_key)
    if cached is not None:
        return cached, True

    raw_data = scrape_reviews(package_name)
    if not raw_data:
        return None, False

    categorized = categorize_by_model(raw_data)
    categorized_dfs = preprocess_categorized(categorized)
    sentiment = sentiment_stats(categorized)

    analysis = {
        "raw_count": len(raw_data),
        "categorized": categorized,
        "categorized_dfs": categorized_dfs,
        "sentiment": sentiment if isinstance(sentiment, dict) else None,
    }
    _cache_set(_ANALYSIS_CACHE, cache_key, analysis)
    return analysis, False


def run_scraping_job(job_id, package_name, query, chat_history):
    try:
        if jobs[job_id].get("cancelled"):
            jobs[job_id]["status"] = "cancelled"
            return

        jobs[job_id]["status"] = "scraping"

        analysis, cache_hit = get_app_analysis(package_name)

        if jobs[job_id].get("cancelled"):
            jobs[job_id]["status"] = "cancelled"
            return

        if not analysis:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"]  = (
                "Link atau package name tidak valid atau aplikasi tidak ditemukan "
                "di Play Store Indonesia. Silakan coba lagi dengan link yang benar, "
                "contoh: https://play.google.com/store/apps/details?id=com.shopee.id"
            )
            return

        jobs[job_id]["status"] = "analyzing" if not cache_hit else "generating"

        categorized         = analysis["categorized"]
        categorized_dfs     = analysis["categorized_dfs"]
        categorized_results = search_categorized(query, categorized_dfs)

        all_relevant = []
        for reviews_list in categorized_results.values():
            all_relevant.extend(reviews_list)

        safe_sentiment = analysis["sentiment"]
        stats          = f"✅ {analysis['raw_count']} data review berhasil diambil"
        if cache_hit:
            stats = f"⚡ Cache dipakai · {analysis['raw_count']} data review siap dipakai"

        if jobs[job_id].get("cancelled"):
            jobs[job_id]["status"] = "cancelled"
            return

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

        if jobs[job_id].get("cancelled"):
            jobs[job_id]["status"] = "cancelled"
            return

        # 4. Simpan hasil
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "query":     query,
            "answer":    str(answer),
            "sentiment": safe_sentiment,
            "token":     token_used,
            "stats":     stats,
            "cache_hit": cache_hit,
            "timestamp": time.time(),
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

    data  = request.get_json(silent=True) or {}
    link  = (data.get("link") or "").strip()
    query = (data.get("query") or "Hi PSAI").strip()
    selected_package_name = normalize_package_name(data.get("package_name") or "")
    package_name = selected_package_name or normalize_package_name(link)

    if not link and not selected_package_name:
        return jsonify({"error": "Masukkan link aplikasi atau cari aplikasi dulu."}), 400

    # Parse package name
    if not selected_package_name and ("http" in link or link.startswith("www.")):
        parsed = urlparse(link if link.startswith("http") else "https://" + link)
        if parsed.netloc not in ("play.google.com", "www.play.google.com"):
            return jsonify({"error": "Link tidak valid. Hanya link aplikasi dari Google Play Store yang diterima. Contoh: https://play.google.com/store/apps/details?id=com.shopee.id"}), 400
        params = parse_qs(parsed.query)
        package_name = params.get("id", [""])[0].split("&")[0].strip()
        if not package_name:
            return jsonify({"error": "Link Play Store tidak mengandung ID aplikasi. Pastikan URL mengandung ?id=nama.paket.aplikasi"}), 400
    elif not package_name or ' ' in package_name or '.' not in package_name or len(package_name) < 5:
        return jsonify({"error": "Input tidak valid. Masukkan link Play Store atau pilih aplikasi dari dropdown."}), 400

    session["package_name"] = package_name
    session.modified = True

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None, "error": None, "cancelled": False}

    t = threading.Thread(
        target=run_scraping_job,
        args=(job_id, package_name, query, list(session["chat_history"])),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id, "package_name": package_name})


@app.route("/api/search_apps", methods=["GET"])
def api_search_apps():
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"results": []})

    cache_key = _make_cache_key("search", q.lower(), "id", "id", 8)
    cached = _cache_get(_SEARCH_CACHE, cache_key)
    if cached is not None:
        return jsonify({"results": cached, "cached": True})

    try:
        apps = search(q, lang="id", country="id", n_hits=8)
        results = []
        for app_item in apps or []:
            app_id = (app_item.get("appId") or "").strip()
            if not app_id:
                # Play Store kadang mengembalikan entri promosi/tanpa appId
                # (mis. kartu developer resmi) — tidak bisa di-scrape, jadi
                # tidak perlu ditampilkan sebagai opsi yang bisa dipilih.
                continue
            results.append({
                "title": app_item.get("title", ""),
                "developer": app_item.get("developer", ""),
                "appId": app_id,
                "icon": app_item.get("icon", ""),
                "url": f"https://play.google.com/store/apps/details?id={app_id}",
            })

        _cache_set(_SEARCH_CACHE, cache_key, results, ttl=900)
        return jsonify({"results": results, "cached": False})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)}), 500


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

    elif job["status"] == "cancelled":
        jobs.pop(job_id, None)

    return jsonify(response)


# ============================================================
# ROUTE: Batalkan job yang sedang berjalan
# ============================================================
@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    job = jobs.get(job_id)
    if job:
        job["cancelled"] = True
    return jsonify({"ok": True})


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
    jobs[job_id] = {"status": "queued", "result": None, "error": None, "cancelled": False}

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