import requests
from textblob import TextBlob
import streamlit as st
import pandas as pd

PETITIONS_API_URL = "https://petition.parliament.uk/petitions.json"


# ---------- DATA HELPERS ---------- #

def fetch_petitions(limit: int = 50):
    """
    Fetch a list of petition titles and summaries from the UK Parliament petitions endpoint.
    Returns a list of dicts with Title, Summary and Combined text (title + background).
    """
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


def analyse_single_text(text: str) -> float:
    """Analyse sentiment for a single text string."""
    return TextBlob(text).sentiment.polarity


# ---------- STREAMLIT APP LAYOUT ---------- #

st.set_page_config(page_title="Gov Data Sentiment Explorer", layout="wide")

st.title("Government Data Sentiment Explorer")

tabs = st.tabs(["Overview", "Petitions API", "CSV Dataset"])

# -------------------- OVERVIEW TAB -------------------- #

with tabs[0]:
    st.subheader("Overview")
    st.write(
        """
        This app analyses sentiment on text from either:

        - the **UK Parliament petitions API**, or  
        - **CSV datasets**, such as public data from **data.gov.uk**.

        For petitions, the analysis is performed on the **petition text itself**
        (the title and its short background summary) – not on individual
        signatures or comments.

        For CSV datasets, each row can represent a separate piece of feedback,
        consultation response or survey answer, allowing you to see overall
        sentiment and drill into individual examples.

        The app uses **TextBlob** for simple sentiment analysis and is intended
        as a small, realistic demo of how lightweight NLP and open data can help
        communicators understand **public mood around a topic**.
        """
    )


# -------------------- PETITIONS API TAB -------------------- #

with tabs[1]:
    st.subheader("Sentiment from UK Parliament Petitions API")

    st.info(
        "This tab analyses the **tone of the petition text itself** "
        "(title + background summary), not the views of individual signatories."
    )

    # Ensure we have a place to store petitions in session state
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
        # Show a preview table
        st.subheader("Petitions preview")
        st.dataframe(
            [
                {"Title": p["Title"], "Summary": p["Summary"]}
                for p in petitions
            ],
            use_container_width=True,
        )

        analysis_mode = st.radio(
            "Analysis mode",
            ["Overall sentiment across fetched petitions", "Analyse a single petition"],
        )

        if analysis_mode == "Overall sentiment across fetched petitions":
            texts = [p["Text"] for p in petitions]
            avg, results = analyse_sentiment(texts)

            if avg > 0:
                overall = "Overall sentiment across these petitions is **slightly positive**."
            elif avg < 0:
                overall = "Overall sentiment across these petitions is **slightly negative**."
            else:
                overall = "Overall sentiment across these petitions is **neutral**."

            st.subheader("Overall sentiment")
            st.markdown(overall)
            st.metric("Average sentiment score", f"{avg:.3f}")

            # Show sample scores per petition
            st.subheader("Sample petitions with sentiment scores")
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

        else:  # Analyse a single petition
            st.subheader("Select a petition to analyse")
            titles = [p["Title"] for p in petitions]

            selected_title = st.selectbox(
                "Choose a petition",
                options=titles,
                index=0 if titles else None,
                key="petition_select",
            )

            if selected_title:
                selected = next(p for p in petitions if p["Title"] == selected_title)
                text = selected["Text"]
                score = analyse_single_text(text)

                if score > 0:
                    overall = "This petition is **slightly positive** in tone."
                elif score < 0:
                    overall = "This petition is **slightly negative** in tone."
                else:
                    overall = "This petition is **neutral** in tone."

                st.subheader("Sentiment for selected petition")
                st.markdown(overall)
                st.metric("Sentiment score", f"{score:.3f}")

                with st.expander("View full petition text"):
                    st.write(f"**Title**: {selected['Title']}")
                    st.write("")
                    st.write(selected["Summary"])


# -------------------- CSV TAB -------------------- #

# -------------------- CSV EXAMPLE DATASETS -------------------- #
# These are predefined CSVs you can load instantly.  
# Replace <your-username> and <your-repo> once your CSV is uploaded to GitHub.

EXAMPLE_CSVS = {
    "Example: NHS Patient Feedback (curated)": {
        "url": "https://raw.githubusercontent.com/Katherine-Holland/sentimentanalysis/refs/heads/main/gov-sentiment-explorer/example_nhs_feedback.csv",
        "text_column": "comment",
    },

    "Demo: Example open CSV": {
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv",
        "text_column": "State",
    },
}

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

    else:  # Example gov/open CSV
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

        # If using an example with a known text column, prefill it
        if chosen_text_column is not None:
            st.info(f"Using text column: **{chosen_text_column}** (from example config)")
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

                if mode_csv_analysis == "Average sentiment (all rows)":
                    avg, _ = analyse_sentiment(texts)
                    if avg > 0:
                        overall = "Overall sentiment is **slightly positive**."
                    elif avg < 0:
                        overall = "Overall sentiment is **slightly negative**."
                    else:
                        overall = "Overall sentiment is **neutral**."
                    st.subheader("Overall sentiment")
                    st.markdown(overall)
                    st.metric("Average sentiment score", f"{avg:.3f}")

                else:
                    st.write("Select a row index to inspect:")

                    selected_idx = st.number_input(
                        "Row index",
                        min_value=0,
                        max_value=len(texts) - 1,
                        value=0,
                        step=1,
                    )
                    selected_text = texts[selected_idx]
                    score = analyse_single_text(selected_text)

                    if score > 0:
                        overall = "This row is **slightly positive** in tone."
                    elif score < 0:
                        overall = "This row is **slightly negative** in tone."
                    else:
                        overall = "This row is **neutral** in tone."

                    st.subheader("Sentiment for selected row")
                    st.markdown(overall)
                    st.metric("Sentiment score", f"{score:.3f}")

                    with st.expander("View full text"):
                        st.write(selected_text)

            except Exception as e:
                st.error("Error analysing CSV text.")
                st.exception(e)
