"""Deteksi indikasi buzzer/fake review berbasis heuristik (bukan model ML).

Clue yang dipakai di sini dilacak ke dua rujukan:

[JAIC] Arfiana D. P., Khothibul Umam, Maya R. Handayani (2025).
  "Identification of Buzzers in Skincare Reviews Using a Lexicon-Based
  Sentiment Analysis Method." JAIC Vol.9 No.5.
  - hal. 2600: buzzer diperkuat lewat "identifikasi pola bahasa yang
    berulang-ulang" -> Rule 2 (near-duplicate) di bawah.
  - hal. 2603: "bagus banget", "cerah seketika", "langsung glowing"
    dianggap buzzer karena "ekspresi berlebihan yang tidak mencerminkan
    opini objektif"; "kata yang terlalu positif dan repetitif ini menjadi
    indikator utama" -> Rule 1 (superlatif) & Rule 3 (hype tanpa detail).
  - hal. 2604: recall model mereka utk kelas buzzer cuma 0.50 (lexicon-based
    ringan memang gampang miss buzzer asli) -> alasan threshold di modul ini
    dibuat konservatif (precision diutamakan, bukan recall).

[Jutisi] Habib Alamsyah, Yana Cahyana, Adi Rizky Pratama (2023).
  "Deteksi Fake Review Menggunakan Metode Support Vector Machine dan
  Naive Bayes Di Tokopedia." Jutisi Vol.12 No.2.
  - Abstrak hal. 585: pola yang sering ditemukan pada fake review adalah
    "penggunaan kata-kata berlebihan dan tidak konsisten dengan pengalaman
    pengguna sebenarnya" -> penguat Rule 1 & Rule 3 (metode mereka ML/TF-IDF,
    dipakai di sini cuma sebagai penguat temuan kualitatifnya, bukan sumber
    aturan baru).

PENTING: modul ini murni informatif — tidak pernah mengubah sentiment_label
atau sentiment_confidence yang sudah dihasilkan IndoBERT di categorize_by_model.
"""

import re

# --- Rule 1: superlatif/hiperbola berlebihan (JAIC hal. 2603) ---
SUPERLATIVE_PHRASES = (
    "banget", "bgt", "sekali", "paling", "terbaik", "luar biasa",
    "the best", "recommended banget", "no debat", "sultan", "juara",
    "mantul", "gercep", "istimewa", "top markotop", "top bgt",
    "sangat sangat", "auto langganan", "wajib punya", "wajib coba",
)

# --- Rule 3: pujian generik tanpa detail spesifik (JAIC hal. 2603 contoh kasus) ---
SPECIFIC_DETAIL_WORDS = (
    "fitur", "menu", "tombol", "loading", "checkout", "chat", "voucher",
    "promo", "cs", "admin", "kirim", "pengiriman", "bayar", "pembayaran",
    "transfer", "saldo", "login", "akun", "update", "versi", "bug",
    "error", "notifikasi", "gambar", "foto", "video", "harga", "produk",
    "pesan", "pesanan", "refund", "komplain", "cod", "ongkir", "kurir",
    "alamat", "password", "verifikasi", "crash", "lag", "lemot", "hp",
    "wifi", "sinyal", "server", "iklan", "spam", "logout", "otp",
)

GENERIC_MAX_WORDS = 20
MIN_SUPERLATIVE_FOR_HYPE = 1

MIN_WORDS_FOR_DUPLICATE_CHECK = 6
DUPLICATE_SIMILARITY_THRESHOLD = 0.7

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text):
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _find_near_duplicate_indices(reviews):
    """Rule 2 — pola bahasa berulang antar ulasan (JAIC hal. 2600).

    Ulasan pendek (<6 kata) sengaja dilewati dari perbandingan ini karena
    pujian generik pendek ("mantap", "bagus banget") wajar diucapkan banyak
    pengguna berbeda secara independen — bukan tanda template/buzzer.
    """
    token_sets = []
    for review in reviews:
        text = str(review.get("content") or "").strip()
        words = text.split()
        token_sets.append(_tokenize(text) if len(words) >= MIN_WORDS_FOR_DUPLICATE_CHECK else None)

    flagged = set()
    n = len(reviews)
    for i in range(n):
        if token_sets[i] is None:
            continue
        for j in range(i + 1, n):
            if token_sets[j] is None:
                continue
            if _jaccard(token_sets[i], token_sets[j]) >= DUPLICATE_SIMILARITY_THRESHOLD:
                flagged.add(i)
                flagged.add(j)

    return flagged


def detect_buzzer_indicators(reviews):
    """Mutasi in-place: nambahin field `buzzer_flag` (bool) dan
    `buzzer_reasons` (list[str]) ke tiap review dict di `reviews`.

    Dipanggil SETELAH categorize_by_model() supaya tiap review sudah punya
    `sentiment_label` — Rule 1+3 (hype berlebihan) cuma berlaku utk ulasan
    yang memang bersentimen POSITIF, soalnya clue di JAIC (hal. 2600) spesifik
    soal "nuansa POSITIF yang berlebihan", bukan kata intensif secara umum
    (tanpa pembatasan ini, keluhan marah pakai huruf kapital + "banget"/
    "sekali" ikut kehitung padahal itu bukan pola buzzer sama sekali).

    Threshold sengaja konservatif — flag cuma dipasang kalau Rule 2
    (duplikat) kena sendiri, ATAU Rule 1 + Rule 3 sama-sama kena. Satu
    sinyal lemah sendirian tidak cukup, supaya ulasan jujur yang memang
    antusias tidak salah ke-flag.
    """
    if not reviews:
        return

    duplicate_idx = _find_near_duplicate_indices(reviews)

    for i, review in enumerate(reviews):
        text = str(review.get("content") or "").strip()
        text_lower = text.lower()
        words = text.split()
        is_positive = review.get("sentiment_label") == "positif"

        reasons = []

        if i in duplicate_idx:
            reasons.append("mirip ulasan lain")

        superlative_score = sum(text_lower.count(p) for p in SUPERLATIVE_PHRASES)
        lacks_detail = not any(w in text_lower for w in SPECIFIC_DETAIL_WORDS)
        is_hype = (
            is_positive
            and superlative_score >= MIN_SUPERLATIVE_FOR_HYPE
            and len(words) <= GENERIC_MAX_WORDS
            and lacks_detail
        )
        if is_hype:
            reasons.append("bahasa berlebihan tanpa detail spesifik")

        review["buzzer_flag"] = bool(reasons)
        review["buzzer_reasons"] = reasons
