def simple_sentiment(relevant_reviews):

    if not relevant_reviews:
        return None

    positive_words = ["bagus", "mantap", "baik", "keren", "recommended", "puas"]
    negative_words = ["jelek", "buruk", "error", "lemot", "parah", "kecewa"]

    score = 0

    for review in relevant_reviews:
        review_lower = review.lower()

        for word in positive_words:
            if word in review_lower:
                score += 1

        for word in negative_words:
            if word in review_lower:
                score -= 1

    if score > 0:
        return "Positif 😊"
    elif score < 0:
        return "Negatif 😞"
    else:
        return "Netral 😐"
