import os
from openai import OpenAI
from dotenv import load_dotenv

print("🚀 LLM MODULE LOADING...")

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ API KEY TIDAK TERDETEKSI")
else:
    print("✅ API KEY TERDETEKSI")

client = OpenAI(api_key=api_key)


def generate_answer(query, relevant_reviews):
    """
    Menghasilkan jawaban AI berdasarkan review relevan.
    Return: (answer, token_used)
    """

    if not relevant_reviews:
        return "Maaf, saya tidak menemukan ulasan yang relevan.", 0

    # 🔥 Batasi review supaya hemat token & cepat
    limited_reviews = relevant_reviews[:5]

    context = "\n\n".join(limited_reviews)

    prompt = f"""
Gunakan hanya ulasan berikut untuk menjawab pertanyaan.

Ulasan:
{context}

Pertanyaan:
{query}

Jawab dengan bahasa natural, profesional, dan ringkas.
"""

    try:
        print("🔥 CALLING OPENAI API...")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Kamu adalah AI analis ulasan aplikasi profesional."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        print("✅ OPENAI RESPONSE RECEIVED")

        answer = response.choices[0].message.content.strip()
        token_used = response.usage.total_tokens

        print("🎯 TOKENS USED:", token_used)

        return answer, token_used

    except Exception as e:
        print("❌ OPENAI ERROR:", e)
        return "Terjadi kesalahan saat memproses AI.", 0
