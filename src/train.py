import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def get_target_column(df):
    if "label" in df.columns:
        return "label"

    if "sentiment" in df.columns:
        return "sentiment"

    raise ValueError("Dataset must contain either a 'label' or 'sentiment' column.")


def train_model(data_path, model_output):
    data_file = Path(data_path)
    model_dir = Path(model_output)

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_file)

    if "clean_text" not in df.columns:
        raise ValueError("Dataset must contain a 'clean_text' column. Run preprocess.py first.")

    target_column = get_target_column(df)

    df = df.dropna(subset=["clean_text", target_column])

    if df.empty:
        raise ValueError("Dataset is empty after removing missing values.")

    x = df["clean_text"].astype(str)
    y = df[target_column]

    if y.nunique() < 2:
        raise ValueError("Target column must contain at least two classes.")

    class_counts = y.value_counts()
    stratify_target = y if class_counts.min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
    )

    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_vectorized, y_train)

    y_pred = model.predict(x_test_vectorized)

    print("Classification Report")
    print(classification_report(y_test, y_pred, zero_division=0))

    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "sentiment_model.joblib"
    vectorizer_path = model_dir / "tfidf_vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Saved model to {model_path}")
    print(f"Saved vectorizer to {vectorizer_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a sentiment classification model."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the cleaned CSV file.",
    )

    parser.add_argument(
        "--model-output",
        "--model_output",
        dest="model_output",
        required=True,
        help="Directory where model files will be saved.",
    )

    args = parser.parse_args()
    train_model(args.data, args.model_output)


if __name__ == "__main__":
    main()
