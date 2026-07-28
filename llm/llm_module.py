import os
import re
import json
import time
import hashlib
from openai import OpenAI

print("🚀 LLM MODULE LOADING...")

_CACHE_TTL_SECONDS = 600
_CONTEXT_CACHE = {}
_STATS_CACHE = {}
_ANSWER_CACHE = {}


def _make_cache_key(*parts):
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(cache, key):
    item = cache.get(key)
    if not item:
        return None

    value, expires_at = item
    if time.time() > expires_at:
        cache.pop(key, None)
        return None

    return value


def _cache_set(cache, key, value, ttl=_CACHE_TTL_SECONDS):
    cache[key] = (value, time.time() + ttl)


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY TIDAK DITEMUKAN DI ENV")
        return None
    return OpenAI(api_key=api_key)


def clean_answer(text):
    if not text:
        return ""
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")
    return text.strip()


SYSTEM_PROMPT = """
Kamu adalah PSAI (PlayStore AI Assistant), asisten analis ulasan aplikasi Google Play Store.

GAYA JAWABAN:
- Jawab natural, mengalir, dan tidak terasa seperti template
- Variasikan cara membuka jawaban; jangan selalu pakai pola kalimat yang sama
- Jangan memaksa format bernomor kecuali memang diminta
- Kalau tidak perlu, jangan ubah semua jawaban jadi poin-poin
- Sesuaikan panjang jawaban dengan pertanyaan user
- Kalau user santai, boleh balas dengan gaya yang lebih santai juga
- Kalau user minta singkat, jawab singkat dan langsung
- Kalau user minta penjelasan, baru perjelas secukupnya
- Kalau user tanya opini, jawab langsung lalu beri alasan singkat yang kuat
- Kalau user tanya analisis, pilih statistik yang relevan lalu simpulkan dengan bahasa yang enak dibaca
- Kalau user tanya hal spesifik, langsung ke inti jawaban tanpa pengantar yang bertele-tele

ATURAN AKURASI:
- Hanya gunakan data ulasan yang diberikan
- Jangan menambah fakta dari luar data
- Kalau datanya tidak cukup, bilang jujur bahwa kesimpulannya belum kuat
- Kalau ada kesimpulan, harus jelas dasarnya dari pola ulasan
- Boleh kutip 1-2 ulasan asli kalau memang memperkuat jawaban
- Kalau user minta ulasan real dengan nama dan timestamp, tampilkan yang memang ada di data dan jangan mengarang identitas atau waktu

FORMAT KUTIPAN ULASAN ASLI:
- Kalau mengutip ulasan asli dari data (dengan nama pengguna, tanggal/timestamp, dan/atau rating yang memang ada di data), WAJIB tulis di baris tersendiri PERSIS dengan format ini, tanda :: harus muncul TEPAT 3 kali, tanpa spasi di sekitar tanda ::, dan wajib ditutup dengan ]:
[ULASAN::Nama Pengguna::Tanggal::Rating::Isi ulasan asli]
- Contoh nyata (rating 4, semua field ada):
[ULASAN::Sari Wulandari::2024-03-12::4::Aplikasinya bagus tapi kadang force close]
- Kalau salah satu dari nama/tanggal/rating tidak ada di data, kosongkan bagian itu tapi tanda :: tetap harus 3 kali. Contoh tanpa tanggal:
[ULASAN::Budi Santoso::::5::Aplikasinya keren banget, gampang dipakai]
- Rating diisi angka 1-5 saja (tanpa kata "rating" atau simbol), kosongkan kalau tidak ada di data
- Jangan pakai format ini untuk parafrase atau ringkasan, hanya untuk kutipan langsung dari teks ulasan asli
- Boleh tulis kalimat analisis biasa sebelum/sesudah baris kutipan ini

ATURAN BAHASA:
- Pakai bahasa Indonesia yang natural
- Jangan pakai backtick atau markdown code block
- Jangan terdengar kaku atau terlalu formal
- Boleh terdengar santai seperlunya, tapi tetap profesional dan berbasis data
- Boleh pakai frasa yang terasa manusiawi seperti "sejauh ini", "yang paling kelihatan", atau "kalau dilihat dari ulasannya"
- Hindari frasa pembuka yang berulang seperti "Berdasarkan data..." di setiap jawaban

JIKA DATA KOSONG:
- Minta user refresh atau ganti link aplikasi
""".strip()


GREETING_RESPONSE = "Halo! Selamat datang di PlayStore AI Assistant (PSAI) 👋"


def build_context(categorized_results, relevant_reviews):
    key = _make_cache_key("context", categorized_results or {}, relevant_reviews or [])
    cached = _cache_get(_CONTEXT_CACHE, key)
    if cached is not None:
        return cached

    def _format_review_item(review):
        if isinstance(review, dict):
            text = str(review.get("review") or review.get("content") or "").strip()
            author = str(review.get("userName") or review.get("author") or review.get("reviewer") or "").strip()
            timestamp = review.get("at") or review.get("timestamp") or review.get("date") or ""
            score = review.get("score")

            head_parts = []
            if author:
                head_parts.append(author)
            if timestamp:
                head_parts.append(str(timestamp))

            head = " | ".join(head_parts)
            if score is not None:
                head = (head + " | " if head else "") + f"rating {score}"

            if head:
                return f"{head}: {text}"
            return text

        return str(review).strip()

    def _limit_reviews(reviews, limit=20, max_len=420):
        out = []
        for review in (reviews or [])[:limit]:
            text = _format_review_item(review)
            if text:
                out.append(text[:max_len])
        return out

    context_parts = []

    good_reviews = _limit_reviews((categorized_results or {}).get("good", []))
    if good_reviews:
        context_parts.append(
            "ULASAN POSITIF:\n" + "\n".join(f"- {r}" for r in good_reviews)
        )

    bad_reviews = _limit_reviews((categorized_results or {}).get("bad", []))
    if bad_reviews:
        context_parts.append(
            "ULASAN NEGATIF:\n" + "\n".join(f"- {r}" for r in bad_reviews)
        )

    if not context_parts:
        result = "\n".join(f"- {r}" for r in _limit_reviews(relevant_reviews, limit=20)) if relevant_reviews else ""
        _cache_set(_CONTEXT_CACHE, key, result)
        return result

    result = "\n\n".join(context_parts)
    _cache_set(_CONTEXT_CACHE, key, result)
    return result


def build_stats_summary(sentiment):
    if not sentiment:
        return ""

    key = _make_cache_key("stats", sentiment)
    cached = _cache_get(_STATS_CACHE, key)
    if cached is not None:
        return cached

    parts = []
    for category, data in sentiment.items():
        label = "Positif" if category == "good" else "Negatif"
        parts.append(f"{label}: {data['count']} ulasan ({data['percentage']}%)")

    result = "STATISTIK ULASAN:\n" + "\n".join(parts)
    _cache_set(_STATS_CACHE, key, result)
    return result


def _build_history_text(chat_history):
    if not chat_history:
        return "-"

    last_turns = chat_history[-4:]
    lines = []
    for turn in last_turns:
        user_text = (turn.get("query") or turn.get("user") or "").strip()
        assistant_text = (turn.get("answer") or turn.get("assistant") or "").strip()
        if user_text:
            lines.append(f"User: {user_text}")
        if assistant_text:
            lines.append(f"Assistant: {assistant_text}")

    return "\n".join(lines) if lines else "-"


def _build_user_prompt(query, context, stats, chat_history):
    history_text = _build_history_text(chat_history)
    return f"""
{stats}

DATA ULASAN:
{context}

RIWAYAT SINGKAT:
{history_text}

PERTANYAAN USER:
{query}

PETUNJUK:
- Jawab sesuai tipe pertanyaan user
- Buat jawaban terasa seperti orang yang sedang menafsirkan ulasan, bukan menyalin format
- Pilih alur jawaban yang paling cocok: ringkas, analitis, atau rekomendatif
- Kalau opini: jawab natural, lalu beri alasan singkat dan jelas
- Kalau analisis: sebut statistik yang relevan lalu simpulkan
- Kalau spesifik: langsung jawab inti pertanyaan
- Kalau user minta rekomendasi, berikan kesimpulan yang jelas di awal lalu dukung dengan alasan
- Kalau ada kutipan ulasan, pilih yang paling relevan dan jangan berlebihan
- Kalau user minta ulasan asli atau real review, boleh kutip nama dan timestamp yang memang tersedia di data
- Kalau data kurang, sampaikan dengan jujur
- Jangan menebak di luar data ulasan yang tersedia
""".strip()


def generate_answer(query, relevant_reviews, categorized_results=None,
                    sentiment=None, chat_history=None, is_first_message=False):
    client = get_client()
    if not client:
        return "OPENAI_API_KEY belum diatur.", 0

    normalized_query = (query or "").strip().lower().replace("👋", "").strip()

    if normalized_query in ("hi psai", "hipsai"):
        return GREETING_RESPONSE, 0

    if not relevant_reviews and not categorized_results:
        return (
            "Maaf, tidak ditemukan ulasan yang relevan dari aplikasi ini. "
            "Silakan coba refresh halaman atau ganti link aplikasi.",
            0
        )

    history_slice = chat_history[-4:] if chat_history else []
    cache_key = _make_cache_key(
        "answer",
        normalized_query,
        relevant_reviews or [],
        categorized_results or {},
        sentiment or {},
        history_slice,
        bool(is_first_message),
    )

    cached = _cache_get(_ANSWER_CACHE, cache_key)
    if cached is not None:
        return cached

    context = build_context(categorized_results or {}, relevant_reviews)
    stats = build_stats_summary(sentiment)
    user_prompt = _build_user_prompt(query, context, stats, chat_history)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=700,
        )

        answer = clean_answer(response.choices[0].message.content or "")
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

        result = (answer, total_tokens)
        _cache_set(_ANSWER_CACHE, cache_key, result, ttl=300)
        return result

    except Exception as e:
        print("❌ OPENAI ERROR:", e)
        return "Terjadi kesalahan saat memproses AI.", 0


def handle_scraping_error():
    return (
        "Maaf, terjadi kesalahan saat mengambil ulasan dari Play Store. "
        "Silakan coba refresh halaman atau ganti link aplikasi."
    )