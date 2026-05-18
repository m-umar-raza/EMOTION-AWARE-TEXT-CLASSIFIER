from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()


def get_vader_score(text):
    scores = analyzer.polarity_scores(str(text))
    return scores["compound"]


def get_vader_label(text):
    score = get_vader_score(text)

    if score >= 0.05:
        return "positive"

    if score <= -0.05:
        return "negative"

    return "neutral"
