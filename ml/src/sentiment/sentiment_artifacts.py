# ml/src/sentiment/artifacts.py

import os
import joblib

# ml/src/sentiment/artifacts.py → ml/models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "models")
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR, "sentiment_tfidf_vectorizer.joblib"
)
MODEL_PATH = os.path.join(
    MODEL_DIR, "sentiment_logistic_regression.joblib"
)


def load_sentiment_artifacts():
    """
    Load sentiment analysis model artifacts.
    This function should be called exactly once per process.
    """

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at {VECTORIZER_PATH}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Sentiment model not found at {MODEL_PATH}"
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    return vectorizer, model
