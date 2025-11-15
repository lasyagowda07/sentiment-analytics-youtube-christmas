import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="John Lewis Ad – YouTube Comment Analysis",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/all_comments_with_topics_and_sentiment.csv")

    # Parse datetime with UTC awareness, then make a plain date column
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["date"] = df["published_at"].dt.date
    else:
        df["date"] = None

    return df

df = load_data()

st.title("John Lewis Christmas Ad – YouTube NLP Analysis")

st.markdown(
    """
This dashboard shows **topic modeling + sentiment analysis** on YouTube comments
for the John Lewis Christmas advert.

Use the filters in the sidebar to explore how audiences reacted.
"""
)

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filters")

# Topic filter
topic_options = ["All"] + sorted(df["topic_label"].dropna().unique().tolist())
chosen_topic = st.sidebar.selectbox("Topic", topic_options)

# Sentiment filter
sentiment_options = ["All"] + sorted(df["sentiment_label"].dropna().unique().tolist())
chosen_sentiment = st.sidebar.selectbox("Sentiment", sentiment_options)

# Date range filter (using df['date'], not the timezone-aware timestamp)
if "date" in df.columns and df["date"].notna().any():
    min_date = df["date"].min()
    max_date = df["date"].max()
    start_date, end_date = st.sidebar.date_input(
        "Date range (published_at)",
        value=(min_date, max_date)
    )
else:
    start_date, end_date = None, None

# =========================
# APPLY FILTERS
# =========================
filtered = df.copy()

if chosen_topic != "All":
    filtered = filtered[filtered["topic_label"] == chosen_topic]

if chosen_sentiment != "All":
    filtered = filtered[filtered["sentiment_label"] == chosen_sentiment]

if start_date and end_date and "date" in filtered.columns:
    filtered = filtered[
        (filtered["date"] >= start_date) &
        (filtered["date"] <= end_date)
    ]

st.write(f"Showing **{len(filtered)}** comments after filters.")

# =========================
# SUMMARY METRICS
# =========================
st.subheader("Summary Metrics")

col1, col2, col3 = st.columns(3)

# Total comments
col1.metric("Total comments", len(filtered))

# Positive / Negative %
if len(filtered) > 0:
    pos_pct = (filtered["sentiment_label"] == "POSITIVE").mean() * 100
    neg_pct = (filtered["sentiment_label"] == "NEGATIVE").mean() * 100
else:
    pos_pct, neg_pct = 0.0, 0.0

col2.metric("Positive %", f"{pos_pct:.1f}%")
col3.metric("Negative %", f"{neg_pct:.1f}%")

# =========================
# TOPIC & SENTIMENT CHARTS
# =========================
st.subheader("Distributions")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Topic distribution (by comments)**")
    if "topic_label" in filtered.columns and len(filtered) > 0:
        topic_counts = filtered["topic_label"].value_counts().sort_values(ascending=False)
        st.bar_chart(topic_counts)
    else:
        st.write("No topic data available for current filters.")

with col_b:
    st.markdown("**Sentiment distribution**")
    if "sentiment_label" in filtered.columns and len(filtered) > 0:
        sent_counts = filtered["sentiment_label"].value_counts()
        st.bar_chart(sent_counts)
    else:
        st.write("No sentiment data available for current filters.")

# =========================
# SENTIMENT OVER TIME
# =========================
if "date" in filtered.columns and filtered["date"].notna().any():
    st.subheader("Sentiment over time (daily)")
    tmp = filtered.copy()
    sent_time = (
        tmp.groupby(["date", "sentiment_label"])
           .size()
           .unstack(fill_value=0)
    )
    if not sent_time.empty:
        st.line_chart(sent_time)
    else:
        st.write("No data to show for selected filters.")

# =========================
# SAMPLE COMMENTS TABLE
# =========================
st.subheader("Sample Comments")

show_cols = [c for c in [
    "author",
    "clean_text",
    "published_at",
    "likes",
    "topic_label",
    "sentiment_label",
    "sentiment_score"
] if c in filtered.columns]

st.dataframe(filtered[show_cols].head(200))

# =========================
# DOWNLOAD FILTERED DATA
# =========================
st.subheader("Download Filtered Data")

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download current view as CSV",
    data=csv_bytes,
    file_name="youtube_comments_filtered.csv",
    mime="text/csv"
)