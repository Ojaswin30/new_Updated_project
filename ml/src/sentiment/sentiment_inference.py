# ml/src/sentiment/inference.py

from typing import List
from src.sentiment.sentiment_artifacts import load_sentiment_artifacts

# load artifacts ONCE at startup
_vectorizer, _model = load_sentiment_artifacts()


def classify_sentiment_batch(texts: List[str]) -> List[str]:
    """
    Classify sentiment for a batch of text inputs.

    Args:
        texts (List[str]): Input text lines

    Returns:
        List[str]: Predicted sentiment labels
    """

    if not texts:
        return []

    cleaned_texts = [
        str(text).strip() for text in texts if str(text).strip()
    ]

    if not cleaned_texts:
        return []

    X = _vectorizer.transform(cleaned_texts)
    predictions = _model.predict(X)

    return predictions.tolist()


if __name__ == "__main__":
    sample_texts = [
        "This product is amazing",
        "Worst purchase ever",
        "Delivery was okay"
    ]

    print(classify_sentiment_batch(sample_texts))
