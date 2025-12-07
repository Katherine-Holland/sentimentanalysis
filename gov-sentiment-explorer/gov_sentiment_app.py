import requests
from textblob import TextBlob
import streamlit as st
import pandas as pd

PETITIONS_API_URL = "https://petition.parliament.uk/petitions.json"


# ---------- DATA HELPERS ---------- #

def fetch_petition_texts(limit: int = 50):
    """
    Fetch a list of petition titles and summaries from the UK Parliament petitions endpoint.
    Returns a list of text strings plus row dicts for display.
    """
    response = requests.get(PETITIONS_API_URL, timeout=10)
    response.raise_for_status()
    data = response.json()

    petitions = data.get("data", [])
    texts = []
    rows = []

    for item in petitions[:limit]:
        attributes = item.get("attributes", {})
        title = attributes.get("action", "") or ""
        summary = attributes.get("background", "") or ""
        combined = f"{title}. {summary}".strip()

        if combined:
            texts.append(combined)
            rows.append({"Title": title, "Summary": summary})

    return texts, rows


def load_csv_from_url(url: str) -> pd.DataFrame:
    """Load a CSV from a URL."""
    return pd.read_csv(url)


def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    """Load a CSV from an uploaded file."""
    return pd.read_csv(uploaded_file)


def extract_text_column(df: pd.DataFrame, column_name: str):
    """Extract clean text values from a column."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found.")
    series = df[column_name].dropna().astype(str)
    return series.tolist()


def analyse_sentiment(texts):
    """Return average sentiment and list of (text, polarity) items."""
    results = [(t, TextBlob(t).sentiment.polarity) for t in texts]

    if not results:
        return 0.0, results

    avg = sum(score for _, score in results) / len(results)
    return avg, results


# ---------- STREAMLIT APP LAYOUT ---------- #

st.set_page_config(page_title="Gov Data Sentiment Explorer", layout="wide")

st.title("Government Data Sentiment Explorer")

tabs = st.tabs(["Overview", "Petitions API", "CSV Dataset"])

# -------------------- OVERVIEW TAB -------------------- #

with tabs[0]:
    st.subheader("Overview")
    st.write(
        """
        I built a Streamlit app that can analyse sentiment on text from either
        the **UK Parliament petitions API** or from **any CSV dataset**, such as a public
        dataset from **data.gov.uk**.

        The app lets a user select a source, loads the data, extracts a chosen
        text column, runs simple sentiment analysis using TextBlob, and reports:

        - overall sentiment  
        - the average polarity score  
        - a sample table of analysed items  

        This is a small, realistic demo of how lightweight NLP and open government
        data can help communicators understand **public mood around a topic**.
        """
    )


# -------------------- PETITIONS API TAB -------------------- #

with tabs[1]:
    st.subheader("Sentiment from UK Parliament Petitions API")

    limit = st.slider("Number of petitions to analyse", 10, 100, 50, step=10)
    run_petitions = st.button("Fetch & analyse petitions")

    if run_petitions:
        try:
            with st.spinner("Fetching petition data..."):
                texts, rows = fetch_petition_texts(limit)
                avg, results = analyse_sentiment(texts)

            if not results:
                st.warning("No text found.")
            else:
                overall = (
                    "Overall sentiment is **slightly positive**."
                    if avg > 0 else
                    "Overall sentiment is **slightly negative**."
                    if avg < 0 else
                    "Overall sentiment is **neutral**."
                )

                st.markdown(overall)
                st.metric("Average sentiment score", f"{avg:.3f}")

                display_rows = [
                    {"Title": row["Title"],
                     "Summary": row["Summary"],
                     "Sentiment score": round(score, 3)}
                    for (_, score), row in zip(results, rows)
                ]

                st.subheader("Sample petitions")
                st.dataframe(display_rows, use_container_width=True)

        except Exception as e:
            st.error("Error fetching or analysing API data.")
            st.exception(e)


# -------------------- CSV TAB -------------------- #

with tabs[2]:
    st.subheader("Analyse Your Own CSV Dataset")

    mode = st.radio("CSV Input Method:", ["From URL", "Upload File"])
    df = None

    if mode == "From URL":
        csv_url = st.text_input("Enter CSV URL")
        if csv_url:
            try:
                df = load_csv_from_url(csv_url)
                st.success("CSV loaded successfully!")
            except Exception as e:
                st.error("Could not load CSV.")
                st.exception(e)

    else:  # Upload
        uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded_file:
            try:
                df = load_csv_from_upload(uploaded_file)
                st.success("CSV loaded successfully!")
            except Exception as e:
                st.error("Could not load uploaded CSV.")
                st.exception(e)

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        column_name = st.text_input("Text column name")
        run_csv = st.button("Analyse CSV")

        if run_csv and column_name:
            try:
                texts = extract_text_column(df, column_name)
                avg, results = analyse_sentiment(texts)

                overall = (
                    "Overall sentiment is **slightly positive**."
                    if avg > 0 else
                    "Overall sentiment is **slightly negative**."
                    if avg < 0 else
                    "Overall sentiment is **neutral**."
                )

                st.markdown(overall)
                st.metric("Average sentiment score", f"{avg:.3f}")

                sample = [
                    {"Text": text[:200] + ("…" if len(text) > 200 else "")}
                    | {"Sentiment score": round(score, 3)}
                    for text, score in results[:100]
                ]

                st.subheader("Sample rows")
                st.dataframe(sample, use_container_width=True)

            except Exception as e:
                st.error("Error analysing CSV text.")
                st.exception(e)
