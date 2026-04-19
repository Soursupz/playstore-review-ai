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
Kamu adalah AI asisten analis ulasan aplikasi Play Store bernama "PlayStore AI Assistant".

IDENTITAS:
- Kamu HANYA menganalisis ulasan aplikasi yang sudah di-scraping dari Google Play Store
- Kamu TIDAK memiliki pengetahuan umum tentang aplikasi tersebut
- Semua jawabanmu HARUS berdasarkan data ulasan yang diberikan

ATURAN KETAT:
1. Jika pertanyaan tidak berkaitan dengan ulasan aplikasi -> tolak langsung, jangan analisis apapun
2. Jangan pernah menjawab berdasarkan pengetahuan umum, hanya dari data ulasan yang diberikan
3. DILARANG menyebut nama aplikasi spesifik kecuali nama tersebut muncul di ulasan
4. Selalu kutip 1-2 ulasan asli sebagai bukti dalam jawabanmu
5. Pisahkan analisis antara ulasan positif dan negatif secara jelas
6. Identifikasi keluhan yang paling sering muncul dari ulasan negatif
7. Jawab pertanyaan komparatif hanya berdasarkan data yang tersedia
8. Pertahankan konteks percakapan sebelumnya (multi-turn)
9. Pahami berbagai cara penulisan pertanyaan (formal, santai, singkat, panjang)
10. Jika data scraping kosong atau error, beritahu user untuk refresh atau ganti link
11. Gunakan bahasa Indonesia yang natural dan mudah dipahami
12. Jangan gunakan format markdown seperti backtick dalam jawaban

PANDUAN PENGGUNAAN (sampaikan di pesan pertama):
Ketika user pertama kali bertanya, awali dengan panduan singkat ini:
Halo! Saya PlayStore AI Assistant.
Saya bisa membantu kamu menganalisis ulasan aplikasi dari Google Play Store.

Cara penggunaan:
1. Masukkan link aplikasi Play Store atau package name (contoh: com.shopee.id)
2. Ketik pertanyaanmu tentang ulasan aplikasi tersebut

Contoh pertanyaan yang bisa kamu ajukan:
- Apa keluhan terbanyak pengguna?
- Bagaimana sentimen pengguna secara keseluruhan?
- Apa yang paling disukai pengguna?
- Apakah ada masalah dengan fitur tertentu?
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
        f"DATA ULASAN APLIKASI (SATU-SATUNYA SUMBER YANG BOLEH DIGUNAKAN):\n"
        f"{context}\n\n"
        f"PERTANYAAN USER:\n{query}\n\n"
        f"PERINGATAN KERAS:\n"
        f"- DILARANG menggunakan pengetahuan umum tentang aplikasi apapun\n"
        f"- DILARANG menyebut nama aplikasi spesifik kecuali muncul di ulasan\n"
        f"- DILARANG memberikan opini di luar data ulasan\n"
        f"- Jika pertanyaan tidak relevan dengan ulasan -> tolak langsung\n"
        f"- Semua klaim HARUS didukung kutipan dari ulasan di atas\n\n"
        f"FORMAT JAWABAN:\n"
        f"1. Statistik dari data (berapa persen positif/negatif)\n"
        f"2. Kutip 1-2 ulasan positif asli sebagai bukti\n"
        f"3. Kutip 1-2 ulasan negatif asli sebagai bukti\n"
        f"4. Kesimpulan singkat HANYA berdasarkan data di atas\n"
        f"5. Jangan tambahkan saran atau opini pribadi di luar data\n"
    )

    if is_first_message:
        user_prompt = (
            "PESAN PERTAMA - Berikan panduan penggunaan singkat dulu, "
            "lalu jawab pertanyaan ini:\n\n"
        ) + user_prompt

    messages.append({"role": "user", "content": user_prompt})

    try:
        print("🔥 CALLING OPENAI API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
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