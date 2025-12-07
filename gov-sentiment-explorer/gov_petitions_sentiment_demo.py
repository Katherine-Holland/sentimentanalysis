import requests
from textblob import TextBlob

# Step 1: Choose an open JSON dataset
# Here we use a sample petitions-style JSON feed as a stand-in for public feedback.
# Replace this URL with a real open data endpoint if desired.
api_url = "https://petition.parliament.uk/petitions.json"

def fetch_petition_texts(limit: int = 50):
    """
    Fetch a list of petition titles and summaries from the UK Parliament petitions endpoint.
    Returns a list of strings to analyse.
    """
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    data = response.json()

    petitions = data.get("data", [])
    texts = []

    for item in petitions[:limit]:
        attributes = item.get("attributes", {})
        title = attributes.get("action", "")
        summary = attributes.get("background", "")
        combined = f"{title}. {summary}".strip()
        if combined:
            texts.append(combined)

    return texts

def analyse_sentiment(texts):
    """
    Run a simple polarity-based sentiment analysis over a list of strings.
    Returns average polarity and a list of individual scores.
    """
    if not texts:
        return 0.0, []

    sentiments = [TextBlob(t).sentiment.polarity for t in texts]
    avg = sum(sentiments) / len(sentiments)
    return avg, sentiments

if __name__ == "__main__":
    try:
        print("Fetching petition texts...")
        texts = fetch_petition_texts(limit=50)

        print(f"Fetched {len(texts)} items. Running sentiment analysis...")
        average_sentiment, scores = analyse_sentiment(texts)

        # Step 4: Summarise the results
        if average_sentiment > 0:
            overall = "Overall sentiment is slightly positive."
        elif average_sentiment < 0:
            overall = "Overall sentiment is slightly negative."
        else:
            overall = "Overall sentiment is neutral."

        print(overall)
        print(f"Average sentiment score: {average_sentiment:.3f}")

    except Exception as e:
        print("An error occurred while analysing sentiment:")
        print(e)
