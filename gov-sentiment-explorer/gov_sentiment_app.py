import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
import pandas as pd

# Download NLTK data if not present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

PETITIONS_API_URL = "https://petition.parliament.uk/petitions.json"


# ---------- DATA HELPERS ---------- #

def fetch_petitions(limit: int = 50):
    response = requests.get(PETITIONS_API_URL, timeout=10)
    response.raise_for_status()
    data = response.json()

    petitions = data.get("data", [])
    items = []

    for item in petitions[:limit]:
        attributes = item.get("attributes", {})
        title = attributes.get("action", "") or ""
        summary = attributes.get("background", "") or ""
        combined = f"{title}. {summary}".strip()

        if combined:
            items.append(
                {
                    "Title": title,
                    "Summary": summary,
                    "Text": combined,
                }
            )
    return items


def load_csv_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def extract_text_column(df: pd.DataFrame, column_name: str):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found.")
    return df[column_name].dropna().astype(str).tolist()


def analyse_sentiment(texts):
    results = []
    for t in texts:
        scores = sia.polarity_scores(t)
        compound = scores["compound"]
        results.append((t, compound))

    if not results:
        return 0.0, results

    avg = sum(score for _, score in results) / len(results)
    return avg, results


def analyse_single_text(text: str):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral/mixed"

    return compound, label


# ---------- STREAMLIT APP LAYOUT ---------- #

st.set_page_config(page_title="Gov Data Sentiment Explorer", layout="wide")

st.title("Government Data Sentiment Explorer")

tabs = st.tabs(["Overview", "Petitions API", "CSV Dataset"])

# -------------------- OVERVIEW TAB -------------------- #

with tabs[0]:
    st.subheader("Overview")
    st.write(
        """
        This app analyses sentiment from:

        - the **UK Parliament petitions API**, and  
        - **CSV datasets** such as open data from **data.gov.uk**.

        For petitions, sentiment is applied to the **petition text itself** 
        (title + background summary).

        For CSV files, each row represents a separate piece of public feedback 
        — useful for consultation responses, surveys or NHS patient comments.

        The app now uses **VADER**, a rule-based sentiment tool optimised for 
        short, emotional text, making it far more accurate than simple polarity 
        tools for public feedback.
        """
    )


# -------------------- PETITIONS API TAB -------------------- #

with tabs[1]:
    st.subheader("Sentiment from UK Parliament Petitions API")

    st.info(
        "This analyses the **tone of the petition text** (title + background), "
        "not the opinions of signatories."
    )

    if "petitions" not in st.session_state:
        st.session_state["petitions"] = []

    limit = st.slider("Number of petitions to fetch", 10, 100, 50, step=10)

    if st.button("Fetch petitions"):
        try:
            with st.spinner("Fetching petition data..."):
                st.session_state["petitions"] = fetch_petitions(limit)

            if not st.session_state["petitions"]:
                st.warning("No petition texts found.")
            else:
                st.success(f"Fetched {len(st.session_state['petitions'])} petitions.")

        except Exception as e:
            st.error("Error fetching petition data.")
            st.exception(e)

    petitions = st.session_state["petitions"]

    if petitions:
        st.subheader("Petitions preview")
        st.dataframe(
            [{"Title": p["Title"], "Summary": p["Summary"]} for p in petitions],
            use_container_width=True,
        )

        analysis_mode = st.radio(
            "Analysis mode",
            ["Overall sentiment across fetched petitions", "Analyse a single petition"],
        )

        if analysis_mode == "Overall sentiment across fetched petitions":
            texts = [p["Text"] for p in petitions]
            avg, results = analyse_sentiment(texts)

            if avg >= 0.05:
                overall = "Overall sentiment is **positive**."
            elif avg <= -0.05:
                overall = "Overall sentiment is **negative**."
            else:
                overall = "Overall sentiment is **neutral/mixed**."

            st.subheader("Overall sentiment")
            st.markdown(overall)
            st.metric("Average sentiment score", f"{avg:.3f}")

            st.subheader("Sample petition scores")
            display_rows = []
            for (text, polarity), p in zip(results, petitions):
                display_rows.append(
                    {
                        "Title": p["Title"],
                        "Summary": p["Summary"],
                        "Sentiment score": round(polarity, 3),
                    }
                )
            st.dataframe(display_rows, use_container_width=True)

        else:
            st.subheader("Select a petition to analyse")
            titles = [p["Title"] for p in petitions]

            selected_title = st.selectbox(
                "Choose a petition",
                options=titles,
                index=0,
            )

            if selected_title:
                selected = next(p for p in petitions if p["Title"] == selected_title)
                text = selected["Text"]

                score, label = analyse_single_text(text)

                if label == "positive":
                    overall = "This petition is **positive** in tone."
                elif label == "negative":
                    overall = "This petition is **negative** in tone."
                else:
                    overall = "This petition is **neutral or mixed** in tone."

                st.subheader("Sentiment for selected petition")
                st.markdown(overall)
                st.metric("Sentiment score", f"{score:.3f}")

                with st.expander("View full petition text"):
                    st.write(f"**Title**: {selected['Title']}")
                    st.write("")
                    st.write(selected["Summary"])


# -------------------- CSV TAB -------------------- #

EXAMPLE_CSVS = {
    "Example: NHS Patient Feedback (curated)": {
        "url": "https://raw.githubusercontent.com/Katherine-Holland/sentimentanalysis/refs/heads/main/gov-sentiment-explorer/example_nhs_feedback.csv",
        "text_column": "comment",
    },

    "Demo: USA states CSV": {
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv",
        "text_column": "State",
    },
}

if "csv_texts" not in st.session_state:
    st.session_state["csv_texts"] = None
if "csv_mode" not in st.session_state:
    st.session_state["csv_mode"] = None
if "csv_avg" not in st.session_state:
    st.session_state["csv_avg"] = None


with tabs[2]:
    st.subheader("Analyse a CSV Dataset")

    mode = st.radio(
        "CSV Input Method:",
        ["From URL", "Upload File", "Example gov/open CSV"],
    )

    df = None
    chosen_text_column = None

    if mode == "From URL":
        csv_url = st.text_input("Enter CSV URL")
        if csv_url:
            try:
                df = load_csv_from_url(csv_url)
                st.success("CSV loaded successfully from URL.")
            except Exception as e:
                st.error("Could not load CSV from URL.")
                st.exception(e)

    elif mode == "Upload File":
        uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded_file:
            try:
                df = load_csv_from_upload(uploaded_file)
                st.success("Uploaded CSV loaded successfully.")
            except Exception as e:
                st.error("Could not load uploaded CSV.")
                st.exception(e)

    else:
        example_name = st.selectbox(
            "Choose an example CSV",
            options=list(EXAMPLE_CSVS.keys()),
        )

        if example_name:
            info = EXAMPLE_CSVS[example_name]
            csv_url = info["url"]
            chosen_text_column = info["text_column"]

            st.write(f"Using example CSV: `{example_name}`")
            st.write(f"Source URL: `{csv_url}`")

            try:
                df = load_csv_from_url(csv_url)
                st.success("Example CSV loaded successfully.")
            except Exception as e:
                st.error("Could not load example CSV.")
                st.exception(e)

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        if chosen_text_column:
            st.info(f"Using text column: **{chosen_text_column}**")
            column_name = chosen_text_column
        else:
            column_name = st.text_input("Text column name")

        mode_csv_analysis = st.radio(
            "Analysis mode",
            ["Average sentiment (all rows)", "Inspect individual rows"],
        )

        run_csv = st.button("Analyse CSV")

        if run_csv and column_name:
            try:
                texts = extract_text_column(df, column_name)
                st.session_state["csv_texts"] = texts
                st.session_state["csv_mode"] = mode_csv_analysis

                if mode_csv_analysis == "Average sentiment (all rows)":
                    avg, _ = analyse_sentiment(texts)
                    st.session_state["csv_avg"] = avg
                else:
                    st.session_state["csv_avg"] = None

            except Exception as e:
                st.error("Error analysing CSV text.")
                st.exception(e)

        texts = st.session_state["csv_texts"]
        stored_mode = st.session_state["csv_mode"]
        stored_avg = st.session_state["csv_avg"]

        if texts is not None and stored_mode is not None:
            if stored_mode == "Average sentiment (all rows)":
                avg = stored_avg if stored_avg is not None else 0
                if avg >= 0.05:
                    overall = "Overall sentiment is **positive**."
                elif avg <= -0.05:
                    overall = "Overall sentiment is **negative**."
                else:
                    overall = "Overall sentiment is **neutral/mixed**."

                st.subheader("Overall sentiment")
                st.markdown(overall)
                st.metric("Average sentiment score", f"{avg:.3f}")

            else:
                st.subheader("Sentiment for selected row")

                selected_idx = st.number_input(
                    "Row index",
                    min_value=0,
                    max_value=len(texts) - 1,
                    value=0,
                    step=1,
                    key="csv_row_index",
                )

                selected_text = texts[int(selected_idx)]
                score, label = analyse_single_text(selected_text)

                if label == "positive":
                    overall = "This row is **positive** in tone."
                elif label == "negative":
                    overall = "This row is **negative** in tone."
                else:
                    overall = "This row is **neutral or mixed** in tone."

                st.markdown(overall)
                st.metric("Sentiment score", f"{score:.3f}")

                with st.expander("View full text"):
                    st.write(selected_text)
