import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "best_sms_spam_model.joblib"
DL_MODEL_DIR = BASE_DIR / "outputs" / "distilbert_sms_spam_model"
CHARTS = [
    ("EDA", BASE_DIR / "outputs" / "01_eda.png"),
    ("Top Words", BASE_DIR / "outputs" / "02_top_words.png"),
    ("Model Comparison", BASE_DIR / "outputs" / "03_model_comparison.png"),
    ("Feature Importance", BASE_DIR / "outputs" / "04_feature_importance.png"),
    ("Dataset Audit", BASE_DIR / "outputs" / "06_dataset_audit.png"),
]

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve",
    "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
    "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "wouldn",
}

SPAM_SIGNAL_PATTERNS = [
    r"\bwon\b",
    r"\bwinner\b",
    r"\blucky\s+(draw|drawn|winner|prize)\b",
    r"\bprize\b",
    r"\bclaim\b",
    r"\bfree\b",
    r"\bcash\b",
    r"\baward(?:ed)?\b",
    r"\bselected\b",
    r"\burgent\b",
    r"\bcall\s+(?:now|me|back)\b",
    r"\btxt\b|\btext\b",
    r"\b\d{5,}\b",
    r"http\S+|www\S+",
]


def spam_signal_score(text: str) -> tuple[int, list[str]]:
    text = text.lower()
    matches = []
    for pattern in SPAM_SIGNAL_PATTERNS:
        if re.search(pattern, text):
            matches.append(pattern)
    return len(matches), matches


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " urltoken ", text)
    text = re.sub(r"\b\d[\d\s]{3,}\d\b", " phonetoken ", text)
    text = re.sub(r"\u00a3|\$|\u20ac", " currencytoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in STOP_WORDS and len(word) > 1]
    return " ".join(tokens)


@st.cache_resource
def load_models():
    models = {}
    if MODEL_PATH.exists():
        models["sklearn"] = {
            "type": "sklearn",
            "name": "Current TF-IDF Linear SVM",
            "model": joblib.load(MODEL_PATH),
        }

    if DL_MODEL_DIR.exists():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(DL_MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(DL_MODEL_DIR))
        model.eval()
        models["distilbert"] = {
            "type": "distilbert",
            "name": "DistilBERT saved model",
            "tokenizer": tokenizer,
            "model": model,
            "torch": torch,
        }

    return models


def predict_sms(message: str, loaded):
    signal_score, signal_matches = spam_signal_score(message)
    if loaded["type"] == "distilbert":
        tokenizer = loaded["tokenizer"]
        model = loaded["model"]
        torch = loaded["torch"]
        inputs = tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        label = "Spam" if prediction == 1 else "Ham"
        if label == "Ham" and signal_score >= 2:
            return "Spam", max(0.85, 1 - confidence), message, signal_matches
        return label, confidence, message, signal_matches

    model = loaded["model"]
    clean_message = preprocess(message)
    prediction = int(model.predict([clean_message])[0])

    if hasattr(model.named_steps["clf"], "predict_proba"):
        probabilities = model.predict_proba([clean_message])[0]
        confidence = float(probabilities[prediction])
    else:
        score = float(model.decision_function([clean_message])[0])
        confidence = float(1 / (1 + np.exp(-abs(score))))

    label = "Spam" if prediction == 1 else "Ham"
    if label == "Ham" and signal_score >= 2:
        return "Spam", max(0.85, 1 - confidence), clean_message, signal_matches
    return label, confidence, clean_message, signal_matches


st.set_page_config(
    page_title="SMS Spam Detector",
    layout="wide",
)

st.title("SMS Spam Detector")
st.caption("Real-time prediction with the current trained model and optional DistilBERT comparison.")

models = load_models()

if not models:
    st.error("Model file not found. Run `python distilbert_sms_spam.py` or `python sms_spam_detection.py` first.")
    st.stop()

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    model_keys = list(models.keys())
    selected_key = st.selectbox(
        "Model",
        model_keys,
        index=model_keys.index("sklearn") if "sklearn" in model_keys else 0,
        format_func=lambda key: models[key]["name"],
    )
    model = models[selected_key]

    if "message" not in st.session_state:
        st.session_state.message = ""

    sample_col, clear_col = st.columns([1, 1])
    with sample_col:
        if st.button("Try spam sample", use_container_width=True):
            st.session_state.message = (
                "WINNER!! You have been selected to receive a GBP 900 prize. Call now!"
            )
    with clear_col:
        if st.button("Try ham sample", use_container_width=True):
            st.session_state.message = "Hey, are you coming to the meeting at 3pm today?"

    message = st.text_area(
        "Message",
        height=180,
        placeholder="Paste an SMS message here...",
        key="message",
    )

    if message.strip():
        label, confidence, clean_message, signal_matches = predict_sms(message, model)
        if label == "Spam":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        st.metric("Confidence", f"{confidence:.2%}")
        with st.expander("Processed text"):
            st.write(clean_message or "(empty after preprocessing)")
        if signal_matches:
            st.caption("Spam signals were found in the message.")
    else:
        st.info("Enter a message to get a prediction.")

with right:
    st.subheader("Model Artifacts")
    st.write(f"Loaded model: `{model['name']}`")
    if selected_key == "distilbert":
        st.warning("This saved DistilBERT model may not match the latest 18k-row dataset unless retrained.")
    st.write("Open the tabs below to review the generated training charts.")

    tabs = st.tabs([title for title, _ in CHARTS])
    for tab, (title, path) in zip(tabs, CHARTS):
        with tab:
            if path.exists():
                st.image(str(path), caption=title, use_container_width=True)
            else:
                st.warning(f"{path.name} has not been generated yet.")
