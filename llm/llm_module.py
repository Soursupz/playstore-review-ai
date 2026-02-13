import os
from openai import OpenAI

print("🚀 LLM MODULE LOADING...")


def get_client():
    """
    Ambil OpenAI client secara aman.
    Tidak membuat client saat module import.
    """

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY TIDAK DITEMUKAN DI ENV")
        return None

    print("✅ OPENAI_API_KEY TERBACA")
    return OpenAI(api_key=api_key)


def generate_answer(query, relevant_reviews):

    client = get_client()

    if not client:
        return "Server belum dikonfigurasi dengan API Key.", 0

    if not relevant_reviews:
        return "Tidak ditemukan ulasan relevan.", 0

    context = "\n\n".join(relevant_reviews)

    prompt = f"""
Gunakan hanya ulasan berikut untuk menjawab pertanyaan.

Ulasan:
{context}

Pertanyaan:
{query}

Jawab dengan bahasa natural dan informatif.
"""

    try:
        print("🔥 CALLING OPENAI API...")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Kamu adalah AI analis ulasan aplikasi."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()
        total_tokens = response.usage.total_tokens

        print("✅ OPENAI RESPONSE RECEIVED")
        print("🎯 TOKENS USED:", total_tokens)

        return answer, total_tokens

    except Exception as e:
        print("❌ OPENAI ERROR:", e)
        return "Terjadi kesalahan saat memproses AI.", 0
