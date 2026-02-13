from google_play_scraper import reviews, Sort

def scrape_reviews(package_name, count=100):
    try:
        result, _ = reviews(
            package_name,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=count
        )

        return result

    except Exception as e:
        print("SCRAPER ERROR:", e)
        return []
