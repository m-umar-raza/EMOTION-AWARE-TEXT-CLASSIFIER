import argparse
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords


def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def clean_text(text, stop_words):
    text = str(text)

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def preprocess_dataset(input_path, output_path):
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_file)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a column named 'text'.")

    stop_words = load_stopwords()

    df["clean_text"] = df["text"].apply(lambda value: clean_text(value, stop_words))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Saved cleaned data to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess text data for sentiment classification."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the cleaned CSV file.",
    )

    args = parser.parse_args()
    preprocess_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
