import os
import re
from openai import OpenAI

print("🚀 LLM MODULE LOADING...")

def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY TIDAK DITEMUKAN DI ENV")
        return None
    print("✅ OPENAI_API_KEY TERBACA")
    return OpenAI(api_key=api_key)

def clean_answer(text):
    if not text:
        return ""
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")
    return text.strip()

SYSTEM_PROMPT = """
Kamu adalah PSAI (PlayStore AI Assistant), asisten analis ulasan aplikasi Google Play Store yang cerdas, komunikatif, dan bisa berpendapat.

IDENTITAS & KEPRIBADIAN:
- Kamu analis berpengalaman yang berani berpendapat secara jujur berdasarkan data
- Kamu berbicara natural seperti teman yang pintar — tidak kaku, tidak selalu pakai format bernomor
- Pendapatmu selalu punya dasar dari pola yang kamu temukan di data ulasan

SUMBER DATA:
- Semua analisis dan pendapat HARUS berdasarkan data ulasan yang diberikan
- DILARANG menggunakan pengetahuan umum tentang aplikasi di luar data ulasan
- DILARANG menyebut nama aplikasi spesifik kecuali muncul di ulasan

TIPE PERTANYAAN & CARA MENJAWAB:

1. PERTANYAAN OPINI ("bagus ga?", "rekomen ga?", "worth it?", "layak download ga?")
   → Jawab langsung dengan pendapatmu (iya/tidak/tergantung) secara natural
   → Jelaskan alasanmu berdasarkan pola di data
   → Dukung dengan 1-2 kutipan ulasan asli sebagai bukti
   → Gaya bahasa santai dan natural, TIDAK pakai format bernomor

2. PERTANYAAN KOMPARATIF/ANALISIS ("sentimen keseluruhan?", "lebih banyak positif atau negatif?", "gimana kondisi aplikasi ini?")
   → Wajib tampilkan statistik (jumlah & persentase positif/negatif)
   → Berikan analisis dan kesimpulan berdasarkan data
   → Boleh kutip ulasan jika memperkuat analisis

3. PERTANYAAN SPESIFIK ("apa keluhan terbanyak?", "fitur apa yang disukai?", "masalah apa yang sering muncul?")
   → Langsung jawab ke poin tanpa statistik
   → Wajib kutip 1-2 ulasan asli sebagai bukti
   → Ringkas dan to the point

ATURAN UMUM:
- Jika pertanyaan tidak berkaitan dengan ulasan/aplikasi → tolak dengan sopan
- Pertahankan konteks percakapan (multi-turn)
- Gunakan bahasa Indonesia yang natural
- Jangan gunakan format markdown seperti backtick
- Jika data kosong → minta user refresh atau ganti link
"""

GREETING_PROMPT = """
Kamu adalah PlayStore AI Assistant (PSAI).
User baru saja menyapamu dengan "Hi PSAI".

Balas dengan sapaan hangat dan ramah, lalu jelaskan cara penggunaan PSAI secara singkat dan jelas.
Gunakan format berikut PERSIS — jangan tambahkan analisis ulasan apapun:

Halo! Selamat datang di PlayStore AI Assistant (PSAI) 👋

Saya siap membantu kamu menganalisis ulasan aplikasi dari Google Play Store menggunakan AI + IndoBERT.

Cara penggunaan:
1. Pastikan kamu sudah memasukkan link Play Store atau package name aplikasi di kolom input
2. Tunggu proses scraping & analisis selesai
3. Setelah selesai, kamu bisa langsung ajukan pertanyaan seperti:
   - Apa keluhan terbanyak pengguna?
   - Bagaimana sentimen pengguna secara keseluruhan?
   - Apa yang paling disukai pengguna?
   - Rekomen ga aplikasi ini?

Silakan mulai bertanya! 😊
"""

def build_context(categorized_results, relevant_reviews):
    context_parts = []

    good_reviews = categorized_results.get("good", [])
    if good_reviews:
        context_parts.append(
            "ULASAN POSITIF:\n" + "\n".join(f'- "{r}"' for r in good_reviews)
        )

    bad_reviews = categorized_results.get("bad", [])
    if bad_reviews:
        context_parts.append(
            "ULASAN NEGATIF:\n" + "\n".join(f'- "{r}"' for r in bad_reviews)
        )

    if not context_parts:
        return "\n".join(f'- "{r}"' for r in relevant_reviews) if relevant_reviews else ""

    return "\n\n".join(context_parts)


def build_stats_summary(sentiment):
    if not sentiment:
        return ""

    parts = []
    for category, data in sentiment.items():
        label = "Positif" if category == "good" else "Negatif"
        parts.append(f"{label}: {data['count']} ulasan ({data['percentage']}%)")

    return "STATISTIK ULASAN:\n" + "\n".join(parts)


def generate_answer(query, relevant_reviews, categorized_results=None,
                    sentiment=None, chat_history=None, is_first_message=False):
    client = get_client()

    if not client:
        return "Server belum dikonfigurasi dengan API Key.", 0

    # ============================================================
    # HANDLER KHUSUS: "Hi PSAI" — hanya balas sapaan & cara pakai
    # ============================================================
    if query.strip().lower().replace("👋", "").strip() in ("hi psai", "hipsai"):
        try:
            print("👋 GREETING MODE: Hi PSAI")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": GREETING_PROMPT},
                    {"role": "user",   "content": "Hi PSAI"}
                ],
                temperature=0.4,
                max_tokens=400
            )
            answer = clean_answer(response.choices[0].message.content)
            tokens = response.usage.total_tokens
            print("✅ GREETING RESPONSE SENT, TOKENS:", tokens)
            return answer, tokens
        except Exception as e:
            print("❌ GREETING ERROR:", e)
            return (
                "Halo! Selamat datang di PlayStore AI Assistant (PSAI) 👋\n\n"
                "Saya siap membantu kamu menganalisis ulasan aplikasi dari Google Play Store.\n\n"
                "Cara penggunaan:\n"
                "1. Masukkan link Play Store atau package name aplikasi\n"
                "2. Tunggu proses scraping & analisis selesai\n"
                "3. Ajukan pertanyaan tentang ulasan aplikasi tersebut\n\n"
                "Silakan mulai bertanya! 😊"
            ), 0

    # ============================================================
    # HANDLER NORMAL: analisis ulasan
    # ============================================================
    if not relevant_reviews and not categorized_results:
        return (
            "Maaf, tidak ditemukan ulasan yang relevan dari aplikasi ini. "
            "Silakan coba refresh halaman atau ganti link aplikasi.", 0
        )

    context = build_context(categorized_results or {}, relevant_reviews)
    stats   = build_stats_summary(sentiment)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for turn in chat_history[-4:]:
            messages.append({"role": "user",      "content": turn.get("query", "")})
            messages.append({"role": "assistant", "content": turn.get("answer", "")})

    user_prompt = (
        f"{stats}\n\n"
        f"DATA ULASAN APLIKASI:\n"
        f"{context}\n\n"
        f"PERTANYAAN USER:\n{query}\n\n"
        f"INSTRUKSI:\n"
        f"Tentukan dulu tipe pertanyaan ini:\n"
        f"- Opini (bagus ga? rekomen ga? worth it?) → jawab langsung dengan pendapat + alasan + 1-2 kutipan bukti, gaya natural\n"
        f"- Komparatif/analisis (sentimen keseluruhan? lebih banyak mana?) → tampilkan statistik + analisis\n"
        f"- Spesifik (keluhan terbanyak? fitur yang disukai?) → langsung ke poin + kutip bukti ulasan\n"
        f"Semua jawaban HARUS berdasarkan data ulasan di atas.\n"
    )

    messages.append({"role": "user", "content": user_prompt})

    try:
        print("🔥 CALLING OPENAI API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=700
        )

        raw_answer   = response.choices[0].message.content
        answer       = clean_answer(raw_answer)
        total_tokens = response.usage.total_tokens

        print("✅ OPENAI RESPONSE RECEIVED")
        print("🎯 TOKENS USED:", total_tokens)

        return answer, total_tokens

    except Exception as e:
        print("❌ OPENAI ERROR:", e)
        return "Terjadi kesalahan saat memproses AI.", 0


def handle_scraping_error():
    return (
        "Maaf, terjadi kesalahan saat mengambil ulasan dari Play Store. "
        "Kemungkinan penyebabnya:\n"
        "1. Link atau package name aplikasi tidak valid\n"
        "2. Aplikasi tidak tersedia di Play Store Indonesia\n"
        "3. Koneksi internet bermasalah\n\n"
        "Silakan coba:\n"
        "- Refresh halaman dan coba lagi\n"
        "- Pastikan link Play Store yang dimasukkan benar\n"
        "- Coba gunakan package name langsung (contoh: com.shopee.id)"
    ), 0