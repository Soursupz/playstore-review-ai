import os
from openai import OpenAI

def generate_answer(query, relevant_reviews):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY tidak ditemukan")
        return "Server belum dikonfigurasi dengan API Key.", 0

    if not relevant_reviews:
        return "Maaf, saya tidak menemukan ulasan yang relevan.", 0

    try:
        client = OpenAI(api_key=api_key)

        context = "\n\n".join(relevant_reviews)

        prompt = f"""
Gunakan hanya ulasan berikut untuk menjawab pertanyaan.

Ulasan:
{context}

Pertanyaan:
{query}

Jawab dengan bahasa natural dan informatif.
"""

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

        usage = response.usage.total_tokens if response.usage else 0

        print("✅ OPENAI RESPONSE RECEIVED")

        return answer, usage

    except Exception as e:
        print("❌ OPENAI ERROR:", e)
        return "Terjadi kesalahan saat memproses AI.", 0
