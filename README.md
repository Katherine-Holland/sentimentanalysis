# UK Gov Open Data – Sentiment Analysis Demo  
*A lightweight NLP tool for analysing public mood using open government datasets.*

This project contains **two components**:

1. **A simple Python script** (`gov_petitions_sentiment_demo.py`)  
2. **A full Streamlit web application** (`gov_sentiment_app.py`)

Both demonstrate how open UK government data can be used to extract insights and help public sector communicators understand audience sentiment around key topics.

---

# Recent Update: Switched from TextBlob → VADER

The project originally used **TextBlob** for polarity scores.  
During testing with real NHS patient feedback, some clearly positive comments were misclassified as slightly negative.

To improve accuracy and reflect real-world government applications, the project now uses:

### **VADER (Valence Aware Dictionary and sEntiment Reasoner)**

VADER is:

- designed for **short, emotional public feedback**
- more accurate for NHS-style comments, surveys and petition text
- better at handling:
  - negation (“not helpful”)
  - intensity (“VERY good”)
  - punctuation (“amazing!!”)
  - case emphasis (“GOOD”)
- widely used in public-sector sentiment analysis

The Streamlit app and helper functions have been upgraded to use VADER’s **compound sentiment score**, which ranges from -1 (negative) to +1 (positive).

---

# Project Overview

This project demonstrates how to:

- Fetch open petition data from the **UK Parliament Petitions API**
- Load external datasets (CSV files) from **data.gov.uk** or local upload
- Extract text fields (titles, summaries, or free-text columns)
- Perform sentiment analysis using **VADER via NLTK**
- Summarise overall public sentiment using:
  - average compound sentiment  
  - positive/neutral/negative classification  
  - per-item sentiment scoring  

In a real Government Communications setting, this pattern can support:

- consultation responses  
- public feedback forms  
- survey free-text fields  
- NHS patient comments  
- campaign impact evaluations  
- audience insights work  
- identifying emerging concerns  
- improving message clarity and targeting  

---

# Components

## 1. Basic Script – `gov_petitions_sentiment_demo.py`

A minimal, self-contained Python script that:

- Pulls live petition data from the UK Parliament JSON endpoint  
- Extracts petition titles + background descriptions  
- Runs sentiment analysis with TextBlob  
- Prints the overall sentiment score and classification  

This is ideal for showing the core pipeline in a simple, readable way.

### **Run the script:**

```bash
pip install requests textblob
python -m textblob.download_corpora
python gov_petitions_sentiment_demo.py

## 2. Streamlit App – `gov_sentiment_app.py`

An interactive web app with **three tabs**:

---

### **Overview**

Explains the purpose of the tool and how it supports public-sector use cases.

---

### **Petitions API**

Fetches real-time petition data from the **UK Parliament API** and:

- analyses sentiment  
- shows average polarity  
- displays a table of petition titles, summaries, and scores  

---

### **CSV Dataset**

Allows users to:

- upload a CSV file **OR** load one from a URL  
- choose a text column  
- run sentiment analysis on free-text survey responses, public feedback, etc.  

This demonstrates real-world flexibility for analysing open government datasets.

---

## Run the Streamlit App

Install dependencies:

```bash
pip install streamlit pandas requests textblob
python -m textblob.download_corpora
streamlit run gov_sentiment_app.py

## Why This Project Matters

Government communicators often work with:

large consultation datasets

public sentiment data

free-text survey responses

feedback from campaigns

This demo shows how a small AI-enabled tool can help:

identify emerging concerns

understand audience tone

support evidence-based communication

improve message clarity and targeting

It reflects the UK Government’s drive toward AI-enabled innovation, in line with the Government Communications Service (GCS) Innovating with Impact strategy.

## Notes

This is a demonstration project for portfolio and learning purposes.
It is not an official government tool.