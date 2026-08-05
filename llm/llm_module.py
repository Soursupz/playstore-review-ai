import os
import re
import json
import time
import hashlib
from openai import OpenAI

from preprocessing.sentiment_cleaning import is_indonesian_text

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


RATING_REQUEST_KEYWORDS = ("rating", "bintang", "star")
_REVIEW_MARKER_RE = re.compile(r"\[ULASAN::([\s\S]*?)\]")


def wants_rating(query):
    q = (query or "").lower()
    return any(kw in q for kw in RATING_REQUEST_KEYWORDS)


def _build_buzzer_lookup(categorized_results, relevant_reviews):
    # Kunci lookup = teks ulasan asli (lowercase, trim) -> status buzzer.
    # SEMUA ulasan dilabeli (bukan cuma yang mencurigakan) — konsisten sama
    # cara kedua jurnal rujukan kerja: tiap komentar diklasifikasi genuine
    # (non-buzzer) atau fake/buzzer, bukan cuma sebagian yang ditandai.
    # Format value: "genuine" atau "fake|alasan1; alasan2".
    lookup = {}

    def _add(reviews):
        for r in reviews or []:
            if not isinstance(r, dict):
                continue
            text = str(r.get("review") or r.get("content") or "").strip().lower()
            if not text:
                continue
            if r.get("buzzer_flag"):
                reasons = "; ".join(r.get("buzzer_reasons") or []) or "berpotensi tidak natural"
                lookup[text] = "fake|" + reasons
            else:
                lookup[text] = "genuine"

    _add((categorized_results or {}).get("good"))
    _add((categorized_results or {}).get("bad"))
    _add(relevant_reviews)
    return lookup


def _match_buzzer_status(marker_text, buzzer_lookup):
    marker_norm = marker_text.strip().lower()
    if not marker_norm or not buzzer_lookup:
        return ""

    if marker_norm in buzzer_lookup:
        return buzzer_lookup[marker_norm]

    # Fallback: konteks yang dikirim ke model dipotong sampai 420 karakter
    # (lihat _limit_reviews), jadi kutipan ulasan panjang bisa jadi cuma
    # cuplikan — dicocokkan lewat containment, bukan exact match saja.
    if len(marker_norm) >= 20:
        for content_norm, status in buzzer_lookup.items():
            if marker_norm in content_norm or content_norm.startswith(marker_norm[:100]):
                return status

    return ""


def postprocess_review_markers(answer, include_rating, buzzer_lookup):
    # Jaring pengaman terakhir + tempat nempelin info yang model sendiri
    # tidak tahu/tidak boleh dipercaya buat mengisi:
    # - Rating kadang tetap diisi model walau instruksi bilang jangan, jadi
    #   dipaksa kosong lagi di sini kalau user tidak eksplisit minta rating.
    # - Status genuine/fake TIDAK PERNAH diminta ke model sama sekali (supaya
    #   tidak bisa dihalusinasi) — backend yang nyisipin sendiri berdasarkan
    #   hasil detect_buzzer_indicators(), dicocokkan lewat teks kutipannya.
    #   SEMUA ulasan yang dikutip dapat status ("genuine" atau "fake|alasan"),
    #   bukan cuma yang mencurigakan.
    # Marker selalu dinormalisasi jadi format 5 field:
    # [ULASAN::Nama::Tanggal::Rating::StatusGenuineFake::Isi]
    def _process_one(match):
        parts = match.group(1).split("::")
        # Model tidak selalu konsisten pakai 4 field persis (kadang placeholder
        # Rating yang kosong malah dihilangkan semua, jadi cuma Nama::Tanggal::Isi)
        # — field diambil selentur mungkin dari depan, sisanya dianggap Isi.
        if len(parts) >= 4:
            name, date, rating = parts[0], parts[1], parts[2]
            text = "::".join(parts[3:])
        elif len(parts) == 3:
            name, date, rating = parts[0], parts[1], ""
            text = parts[2]
        elif len(parts) == 2:
            name, date, rating = parts[0], "", ""
            text = parts[1]
        else:
            return match.group(0)

        if not include_rating:
            rating = ""

        flag = _match_buzzer_status(text, buzzer_lookup)

        return "[ULASAN::" + "::".join([name, date, rating, flag, text]) + "]"

    return _REVIEW_MARKER_RE.sub(_process_one, answer or "")


SYSTEM_PROMPT = """
Kamu adalah ASAI (App Store AI Assistant), asisten analis ulasan aplikasi Apple App Store.

BATASAN TOPIK (WAJIB DIPATUHI PALING UTAMA):
- Kamu HANYA boleh membahas ulasan aplikasi yang sedang dianalisis: keluhan, pujian, sentimen, rating, rekomendasi, dan hal terkait pengalaman pengguna aplikasi ini
- Kalau pertanyaan user TIDAK berhubungan dengan ulasan/aplikasi yang sedang dibahas (misalnya tanya tokoh publik, politik, pengetahuan umum, matematika, coding, curhat pribadi, dll), JANGAN dijawab isinya sama sekali
- Untuk pertanyaan di luar topik, cukup minta maaf singkat dan jelaskan kamu cuma bisa bantu soal ulasan aplikasi ini, lalu ajak user balik ke topik. Jangan tetap memberi jawaban informatif tentang topik di luar konteks itu meskipun kamu tahu jawabannya
- Kalau pertanyaan ambigu (mungkin terkait mungkin tidak), tanyakan klarifikasi singkat daripada langsung berasumsi dan menjawab di luar topik

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

SUMBER KEBENARAN SENTIMEN (WAJIB DIPATUHI):
- Data ulasan sudah dikelompokkan jadi ULASAN POSITIF dan ULASAN NEGATIF berdasarkan hasil klasifikasi model sentimen (IndoBERT) terhadap TEKS ulasan, bukan berdasarkan rating bintang
- Rating bintang cuma info tambahan yang ditulis user sendiri, seringkali TIDAK sinkron dengan isi teksnya (mis. teks isinya positif tapi kasih 1 bintang karena kecewa soal hal lain di luar aplikasi, atau sebaliknya)
- Kalau nentuin/nyebut suatu ulasan itu positif atau negatif, SELALU ikuti kelompoknya (ULASAN POSITIF/ULASAN NEGATIF) sesuai klasifikasi teks, JANGAN menyimpulkan sentimen dari angka rating semata
- Jangan heran atau mengoreksi kalau ada ulasan rating rendah masuk kelompok positif (atau sebaliknya) — itu wajar karena klasifikasi berbasis makna teks, bukan angka bintang. Jangan menyebutnya sebagai kejanggalan atau kesalahan data

FORMAT KUTIPAN ULASAN ASLI:
- Kalau mengutip ulasan asli dari data (dengan nama pengguna dan/atau tanggal/timestamp yang memang ada di data), WAJIB tulis di baris tersendiri PERSIS dengan format ini, tanda :: harus muncul TEPAT 3 kali, tanpa spasi di sekitar tanda ::, dan wajib ditutup dengan ]:
[ULASAN::Nama Pengguna::Tanggal::Rating::Isi ulasan asli]
- ATURAN RATING/BINTANG (WAJIB, JANGAN DILANGGAR): field Rating HARUS DIKOSONGKAN SECARA DEFAULT walaupun rating aslinya ADA di data. Field Rating HANYA boleh diisi kalau user di pertanyaannya SECARA EKSPLISIT minta lihat rating/bintang (mis. "tampilkan rating juga", "berapa bintangnya", "sekalian kasih liat ratingnya"). Kalau user cuma bilang "tampilkan ulasan asli"/"kasih contoh review"/"ada ulasan apa aja" TANPA menyebut kata rating/bintang, field Rating WAJIB kosong — jangan diisi meski datanya tersedia
- Contoh default (user TIDAK minta rating, field Rating dikosongkan walau di data ada ratingnya):
[ULASAN::Sari Wulandari::2024-03-12::::Aplikasinya bagus tapi kadang force close]
- Contoh kalau user EKSPLISIT minta rating/bintang:
[ULASAN::Sari Wulandari::2024-03-12::4::Aplikasinya bagus tapi kadang force close]
- Rating diisi angka 1-5 saja (tanpa kata "rating" atau simbol) kalau memang diminta
- Kalau nama/tanggal juga tidak ada di data, kosongkan juga bagian itu, tapi tanda :: tetap harus 3 kali
- Jangan pakai format ini untuk parafrase atau ringkasan, hanya untuk kutipan langsung dari teks ulasan asli
- Boleh tulis kalimat analisis biasa sebelum/sesudah baris kutipan ini

KALAU USER MINTA ULASAN REAL/ASLI (WAJIB, TIDAK BOLEH DILANGGAR):
- Ini berlaku untuk permintaan apapun bentuknya: "tampilkan ulasan real/asli", "kasih contoh ulasan", "ada ulasan apa aja", "tunjukkan review positif dan negatif", dsb — baik diminta satu maupun banyak sekaligus
- SETIAP ulasan yang ditampilkan WAJIB pakai baris marker [ULASAN::Nama::Tanggal::Rating::Isi], satu marker untuk satu ulasan, masing-masing di baris sendiri
- INGAT LAGI: field Rating di marker-marker ini WAJIB kosong kecuali user eksplisit minta rating/bintang ditampilkan (lihat ATURAN RATING/BINTANG di atas) — jangan otomatis isi rating cuma karena lagi menampilkan banyak ulasan sekaligus
- DILARANG membuat heading markdown seperti **Ulasan Positif:** atau **Ulasan Negatif:**, dan DILARANG membuat penomoran manual seperti "1. Nama, dengan rating X, mengungkapkan..." — itu bukan kutipan, itu parafrase berbalut format, dan TIDAK akan tampil sebagai card ke user
- Kalau perlu memisahkan kelompok positif dan negatif, cukup pakai satu kalimat natural biasa (tanpa bold/heading) sebagai pengantar sebelum kumpulan marker, misalnya "Untuk yang positif, ada beberapa yang bilang:" lalu langsung deretan marker
- Isi field "Isi" pada marker HARUS teks asli dari data, bukan ringkasan atau parafrase kalimat user

ATURAN BAHASA:
- Pakai bahasa Indonesia yang natural
- Jangan pakai backtick, markdown code block, bold (**teks**), atau heading markdown (## dsb) — tampilan chat tidak merender markdown jadi itu akan muncul sebagai simbol mentah
- Jangan terdengar kaku atau terlalu formal
- Boleh terdengar santai seperlunya, tapi tetap profesional dan berbasis data
- Boleh pakai frasa yang terasa manusiawi seperti "sejauh ini", "yang paling kelihatan", atau "kalau dilihat dari ulasannya"
- Hindari frasa pembuka yang berulang seperti "Berdasarkan data..." di setiap jawaban

JIKA DATA KOSONG:
- Minta user refresh atau ganti link aplikasi
""".strip()


GREETING_RESPONSE = "Halo! Selamat datang di App Store AI Assistant (ASAI) 👋"


def _review_text(review):
    if isinstance(review, dict):
        return str(review.get("review") or review.get("content") or "").strip()
    return str(review).strip()


def _filter_indonesian_reviews(reviews):
    # Lapisan pertahanan kedua terhadap ulasan berbahasa asing — dijalankan
    # lagi di sini (bukan cuma pas scraping) supaya data lama yang sempat
    # ke-cache sebelum filter bahasa dipasang tetap ikut tersaring.
    out = []
    for review in reviews or []:
        text = _review_text(review)
        if not text or is_indonesian_text(text):
            out.append(review)
    return out


def build_context(categorized_results, relevant_reviews, include_rating=False):
    key = _make_cache_key("context", categorized_results or {}, relevant_reviews or [], bool(include_rating))
    cached = _cache_get(_CONTEXT_CACHE, key)
    if cached is not None:
        return cached

    def _format_review_item(review):
        if isinstance(review, dict):
            text = _review_text(review)
            author = str(review.get("userName") or review.get("author") or review.get("reviewer") or "").strip()
            timestamp = review.get("at") or review.get("timestamp") or review.get("date") or ""
            score = review.get("score")

            head_parts = []
            if author:
                head_parts.append(author)
            if timestamp:
                head_parts.append(str(timestamp))

            head = " | ".join(head_parts)
            # Rating cuma dimasukkan ke konteks kalau user memang minta —
            # kalau selalu disertakan, model jadi ikut nyebut rating di
            # jawabannya walau instruksi output bilang jangan.
            if include_rating and score is not None:
                head = (head + " | " if head else "") + f"rating {score}"

            if head:
                return f"{head}: {text}"
            return text

        return str(review).strip()

    def _limit_reviews(reviews, limit=20, max_len=420):
        out = []
        for review in _filter_indonesian_reviews(reviews)[:limit]:
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
- Kalau user minta ulasan asli atau real review (dalam bentuk apapun), WAJIB tampilkan tiap ulasan dengan marker [ULASAN::Nama::Tanggal::Rating::Isi] satu per baris — jangan pakai heading bold atau penomoran manual
- Kalau data kurang, sampaikan dengan jujur
- Jangan menebak di luar data ulasan yang tersedia
""".strip()


def generate_answer(query, relevant_reviews, categorized_results=None,
                    sentiment=None, chat_history=None, is_first_message=False):
    client = get_client()
    if not client:
        return "OPENAI_API_KEY belum diatur.", 0

    normalized_query = (query or "").strip().lower().replace("👋", "").strip()

    if normalized_query in ("hi asai", "hiasai"):
        return GREETING_RESPONSE, 0

    if not relevant_reviews and not categorized_results:
        return (
            "Maaf, tidak ditemukan ulasan yang relevan dari aplikasi ini. "
            "Silakan coba refresh halaman atau ganti link aplikasi.",
            0
        )

    include_rating = wants_rating(query)

    history_slice = chat_history[-4:] if chat_history else []
    cache_key = _make_cache_key(
        "answer",
        normalized_query,
        relevant_reviews or [],
        categorized_results or {},
        sentiment or {},
        history_slice,
        bool(is_first_message),
        include_rating,
    )

    cached = _cache_get(_ANSWER_CACHE, cache_key)
    if cached is not None:
        return cached

    context = build_context(categorized_results or {}, relevant_reviews, include_rating=include_rating)
    stats = build_stats_summary(sentiment)
    user_prompt = _build_user_prompt(query, context, stats, chat_history)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=700,
        )

        answer = clean_answer(response.choices[0].message.content or "")
        buzzer_lookup = _build_buzzer_lookup(categorized_results, relevant_reviews)
        answer = postprocess_review_markers(answer, include_rating, buzzer_lookup)

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
        "Maaf, terjadi kesalahan saat mengambil ulasan dari App Store. "
        "Silakan coba refresh halaman atau ganti link aplikasi."
    )