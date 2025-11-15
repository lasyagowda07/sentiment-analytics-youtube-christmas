import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (safe for scripts/servers)
import matplotlib.pyplot as plt


# ---------- CONFIG ----------

INPUT_CSV = "all_comments_with_topics_and_sentiment.csv"
OUTPUT_DIR = Path("charts")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------- HELPER: LOAD & PREP DATA ----------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Parse datetime if present
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["date"] = df["published_at"].dt.date
    else:
        df["date"] = pd.NaT

    return df


# ---------- PLOT 1: OVERALL SENTIMENT ----------

def plot_overall_sentiment(df: pd.DataFrame, out_path: Path) -> None:
    if "sentiment_label" not in df.columns:
        print("No 'sentiment_label' column found. Skipping overall sentiment chart.")
        return

    counts = df["sentiment_label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Overall Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Comments")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved overall sentiment chart -> {out_path}")


# ---------- PLOT 2: TOPIC DISTRIBUTION ----------

def plot_topic_distribution(df: pd.DataFrame, out_path: Path) -> None:
    if "topic_label" not in df.columns:
        print("No 'topic_label' column found. Skipping topic distribution chart.")
        return

    topic_counts = df["topic_label"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    topic_counts.plot(kind="bar")
    plt.title("Topic Distribution (Share of Comments per Topic)")
    plt.xlabel("Topic")
    plt.ylabel("Number of Comments")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved topic distribution chart -> {out_path}")


# ---------- PLOT 3: SENTIMENT BY TOPIC (STACKED BAR) ----------

def plot_sentiment_by_topic(df: pd.DataFrame, out_path: Path) -> None:
    if "topic_label" not in df.columns or "sentiment_label" not in df.columns:
        print("Missing 'topic_label' or 'sentiment_label'. Skipping sentiment-by-topic chart.")
        return

    # Pivot table: rows = topic, cols = sentiment, values = counts
    crosstab = pd.crosstab(df["topic_label"], df["sentiment_label"])
    # Convert to percentages by row
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(8, 5))
    bottom = None
    for sentiment in sorted(crosstab_pct.columns):
        values = crosstab_pct[sentiment]
        if bottom is None:
            plt.bar(crosstab_pct.index, values, label=sentiment)
            bottom = values
        else:
            plt.bar(crosstab_pct.index, values, bottom=bottom, label=sentiment)
            bottom = bottom + values

    plt.title("Sentiment within Each Topic (Stacked %)")
    plt.xlabel("Topic")
    plt.ylabel("Percentage of Comments")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved sentiment-by-topic chart -> {out_path}")


# ---------- PLOT 4: COMMENTS PER DAY ----------

def plot_comments_per_day(df: pd.DataFrame, out_path: Path) -> None:
    if "date" not in df.columns or df["date"].isna().all():
        print("No 'date' column or all dates NaT. Skipping comments-per-day chart.")
        return

    counts = df.groupby("date").size()

    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.title("Number of Comments per Day")
    plt.xlabel("Date")
    plt.ylabel("Number of Comments")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved comments-per-day chart -> {out_path}")


# ---------- PLOT 5: SENTIMENT OVER TIME (LINE) ----------

def plot_sentiment_over_time(df: pd.DataFrame, out_path: Path) -> None:
    if "date" not in df.columns or df["date"].isna().all() or "sentiment_label" not in df.columns:
        print("Missing 'date' or 'sentiment_label'. Skipping sentiment-over-time chart.")
        return

    tmp = df.copy()
    # Group by date and sentiment, then unstack to get columns POSITIVE/NEGATIVE
    counts = (
        tmp.groupby(["date", "sentiment_label"])
           .size()
           .unstack(fill_value=0)
           .sort_index()
    )

    if counts.empty:
        print("No data after grouping for sentiment-over-time. Skipping chart.")
        return

    plt.figure(figsize=(8, 4))
    for col in counts.columns:
        plt.plot(counts.index, counts[col], marker="o", label=col)

    plt.title("Sentiment Over Time (Daily Counts)")
    plt.xlabel("Date")
    plt.ylabel("Number of Comments")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved sentiment-over-time chart -> {out_path}")


# ---------- MAIN ----------

def main():
    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f"Could not find {INPUT_CSV} in current directory.")

    df = load_data(INPUT_CSV)

    # 1. Overall sentiment
    plot_overall_sentiment(df, OUTPUT_DIR / "overall_sentiment.png")

    # 2. Topic distribution
    plot_topic_distribution(df, OUTPUT_DIR / "topic_distribution.png")

    # 3. Sentiment within each topic (stacked bar)
    plot_sentiment_by_topic(df, OUTPUT_DIR / "sentiment_by_topic.png")

    # 4. Comments per day
    plot_comments_per_day(df, OUTPUT_DIR / "comments_per_day.png")

    # 5. Sentiment over time
    plot_sentiment_over_time(df, OUTPUT_DIR / "sentiment_over_time.png")


if __name__ == "__main__":
    main()