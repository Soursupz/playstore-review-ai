def sentiment_stats(categorized):
    """
    Hitung jumlah review per kategori dan persentasenya.
    categorized = {"bad": [...], "good": [...]}
    """
    total = sum(len(v) for v in categorized.values())
    if total == 0:
        return None

    stats = {}
    for category, reviews_list in categorized.items():
        count = len(reviews_list)
        pct   = round((count / total) * 100, 1)
        stats[category] = {"count": count, "percentage": pct}

    return stats