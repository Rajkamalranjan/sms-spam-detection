"""
DistilBERT SMS spam classifier.

Train:
    python distilbert_sms_spam.py

Predict:
    python distilbert_sms_spam.py --predict "Your SMS message here"
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "SMSSpamCollection.txt"
EXTRA_DATA_DIR = BASE_DIR / "datasets"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUT_DIR / "distilbert_sms_spam_model"
METRICS_PATH = OUT_DIR / "distilbert_metrics.json"
CHART_PATH = OUT_DIR / "05_distilbert_evaluation.png"
PREDICTIONS_PATH = OUT_DIR / "distilbert_test_predictions.csv"

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 128
RANDOM_STATE = 42
BALANCE_TRAINING_DATA = True
LABEL_MAP = {
    "ham": 0,
    "not_spam": 0,
    "not spam": 0,
    "legit": 0,
    "normal": 0,
    "safe": 0,
    "0": 0,
    0: 0,
    "spam": 1,
    "phishing": 1,
    "scam": 1,
    "fraud": 1,
    "promo": 1,
    "promotion": 1,
    "junk": 1,
    "1": 1,
    1: 1,
}
TEXT_COLUMNS = ("message", "text", "body", "content", "tweet", "email", "subject")
LABEL_COLUMNS = ("label", "class", "category", "target", "is_spam", "spam")
BUILTIN_MULTICHANNEL_EXAMPLES = [
    # Email-style spam/phishing
    ("spam", "URGENT: Your mailbox storage is full. Verify your account now at http://mail-security.example/verify", "synthetic_email"),
    ("spam", "Congratulations, you have been selected for a $1000 gift card. Claim your reward today.", "synthetic_email"),
    ("spam", "Invoice payment failed. Open the attached billing portal and update your card immediately.", "synthetic_email"),
    ("spam", "Security alert: unusual login detected. Confirm your password to avoid suspension.", "synthetic_email"),
    ("spam", "Limited time crypto bonus. Deposit now and double your money in 24 hours.", "synthetic_email"),
    ("spam", "Your package delivery is on hold. Pay a small redelivery fee at the link below.", "synthetic_email"),
    ("spam", "Loan approved with zero paperwork. Reply YES to get instant cash transfer.", "synthetic_email"),
    ("spam", "Final notice from support team: your account will be closed unless you validate today.", "synthetic_email"),
    ("ham", "Can you review the project notes before our 3 PM meeting?", "synthetic_email"),
    ("ham", "The updated report is attached. Please share feedback when you get time.", "synthetic_email"),
    ("ham", "Your order has shipped and should arrive on Monday. Tracking details are in your account.", "synthetic_email"),
    ("ham", "Thanks for joining the interview call today. We will send the next steps soon.", "synthetic_email"),
    ("ham", "Reminder: your dentist appointment is scheduled for Friday at 10 AM.", "synthetic_email"),
    ("ham", "I pushed the latest code changes. Please pull before starting your work.", "synthetic_email"),
    ("ham", "Here is the agenda for tomorrow's team sync.", "synthetic_email"),
    ("ham", "Your monthly bank statement is ready to view in the official app.", "synthetic_email"),
    # Twitter/X, DM, and social-message style examples
    ("spam", "You won our giveaway! DM your phone number and click this link to claim.", "synthetic_social"),
    ("spam", "Free followers for first 500 users. Retweet and enter your password here.", "synthetic_social"),
    ("spam", "I made $700 today from home. Message me NOW for the secret method.", "synthetic_social"),
    ("spam", "Your profile will be verified today. Send login code to continue.", "synthetic_social"),
    ("spam", "Hot deal: iPhone 15 for only $49. Click before it expires.", "synthetic_social"),
    ("spam", "Investment group open for 1 hour only. Guaranteed profit, no risk.", "synthetic_social"),
    ("spam", "We noticed copyright violation on your account. Appeal through this short link.", "synthetic_social"),
    ("spam", "Private video leaked. Tap here to watch before it is deleted.", "synthetic_social"),
    ("ham", "Loved your thread about machine learning datasets. Very useful examples.", "synthetic_social"),
    ("ham", "Are you joining the space tonight or should I send notes later?", "synthetic_social"),
    ("ham", "Happy birthday! Hope your day is excellent.", "synthetic_social"),
    ("ham", "The event photos look great. Thanks for sharing them.", "synthetic_social"),
    ("ham", "Can you DM me the restaurant address?", "synthetic_social"),
    ("ham", "Nice work on the new release. The dashboard feels faster now.", "synthetic_social"),
    ("ham", "Poll closes at 6 PM. Vote when you get a chance.", "synthetic_social"),
    ("ham", "I will be offline for an hour, ping me after lunch.", "synthetic_social"),
    # WhatsApp/Telegram/promo-like short messages
    ("spam", "KYC pending. Your wallet will be blocked today. Complete verification at the link.", "synthetic_chat"),
    ("spam", "Dear customer, you are eligible for a free recharge. Send OTP to receive benefit.", "synthetic_chat"),
    ("spam", "Win cash daily. Join our betting group and get sure-shot tips.", "synthetic_chat"),
    ("spam", "Your ATM card is suspended. Call support immediately to reactivate.", "synthetic_chat"),
    ("spam", "Flash sale voucher unlocked. Click to claim before midnight.", "synthetic_chat"),
    ("spam", "Part-time job: earn 5000 per day by liking videos. Register now.", "synthetic_chat"),
    ("ham", "Reach home and call me once you are free.", "synthetic_chat"),
    ("ham", "Meeting shifted to 11:30. Please update the calendar.", "synthetic_chat"),
    ("ham", "I transferred the rent. Check and confirm.", "synthetic_chat"),
    ("ham", "Please bring milk and bread on the way back.", "synthetic_chat"),
    ("ham", "Class notes are in the group drive folder.", "synthetic_chat"),
    ("ham", "Your cab is waiting near gate number two.", "synthetic_chat"),
]
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


def spam_signal_score(text):
    text = text.lower()
    matches = []
    for pattern in SPAM_SIGNAL_PATTERNS:
        if re.search(pattern, text):
            matches.append(pattern)
    return len(matches), matches


class SmsDataset(Dataset):
    def __init__(self, messages, labels, tokenizer):
        self.encodings = tokenizer(
            list(messages),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def normalize_label(value):
    if pd.isna(value):
        return None
    key = value.strip().lower() if isinstance(value, str) else value
    return LABEL_MAP.get(key)


def normalize_message(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def build_dataset(records):
    df = pd.DataFrame(records, columns=["label", "message", "source"])
    df["message"] = df["message"].map(normalize_message)
    df["label_num"] = df["label"].map(normalize_label)
    df = df.dropna(subset=["label_num"])
    df = df[df["message"].str.len() > 0].copy()
    df["label_num"] = df["label_num"].astype(int)
    df["label"] = df["label_num"].map({0: "ham", 1: "spam"})
    return df[["label", "message", "label_num", "source"]]


def load_sms_collection():
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )
    df["source"] = "sms_spam_collection"
    return build_dataset(df[["label", "message", "source"]].to_records(index=False))


def builtin_multichannel_data():
    return build_dataset(BUILTIN_MULTICHANNEL_EXAMPLES)


def read_table(path):
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)

    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
    delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter if sample else ","
    return pd.read_csv(path, sep=delimiter, encoding="utf-8")


def load_extra_dataset_file(path):
    raw = read_table(path)
    lower_columns = {str(column).strip().lower(): column for column in raw.columns}
    label_column = next((lower_columns[name] for name in LABEL_COLUMNS if name in lower_columns), None)
    text_columns = [lower_columns[name] for name in TEXT_COLUMNS if name in lower_columns]

    if label_column is None or not text_columns:
        print(f"Skipping {path.name}: need one label column and one text/message/body column.")
        return pd.DataFrame(columns=["label", "message", "label_num", "source"])

    text = raw[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    records = pd.DataFrame(
        {
            "label": raw[label_column],
            "message": text,
            "source": path.stem,
        }
    )
    return build_dataset(records.to_records(index=False))


def load_extra_datasets():
    if not EXTRA_DATA_DIR.exists():
        return pd.DataFrame(columns=["label", "message", "label_num", "source"])

    frames = []
    for path in sorted(EXTRA_DATA_DIR.iterdir()):
        if path.suffix.lower() in {".csv", ".tsv", ".txt", ".json", ".jsonl"}:
            frames.append(load_extra_dataset_file(path))

    if not frames:
        return pd.DataFrame(columns=["label", "message", "label_num", "source"])
    return pd.concat(frames, ignore_index=True)


def load_data(include_builtin=True, include_extra=True):
    frames = [load_sms_collection()]
    if include_builtin:
        frames.append(builtin_multichannel_data())
    if include_extra:
        frames.append(load_extra_datasets())

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["message", "label_num"]).reset_index(drop=True)
    source_counts = df["source"].value_counts().sort_index().to_dict()
    return df, before - len(df), source_counts


def label_distribution(df):
    counts = df["label"].value_counts().sort_index()
    return {label: int(count) for label, count in counts.items()}


def source_label_distribution(df):
    grouped = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    return {
        source: {label: int(count) for label, count in row.items()}
        for source, row in grouped.sort_index().iterrows()
    }


def balance_training_data(train_df):
    min_class_count = train_df["label_num"].value_counts().min()
    balanced = (
        train_df.groupby("label_num", group_keys=False)
        .sample(n=min_class_count, random_state=RANDOM_STATE)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    return balanced


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def train_model():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, duplicate_count, source_counts = load_data()

    train_df_raw, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label_num"],
    )
    train_df = balance_training_data(train_df_raw) if BALANCE_TRAINING_DATA else train_df_raw

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1},
    )

    train_dataset = SmsDataset(train_df["message"], train_df["label_num"], tokenizer)
    test_dataset = SmsDataset(test_df["message"], test_df["label_num"], tokenizer)

    counts = train_df["label_num"].value_counts().sort_index().to_numpy()
    class_weights = counts.sum() / (len(counts) * counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "distilbert_checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=25,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("=" * 60)
    print("  DISTILBERT SMS SPAM TRAINING")
    print("=" * 60)
    print(f"Messages after duplicate removal : {len(df)}")
    print(f"Duplicates removed               : {duplicate_count}")
    print(f"Raw train messages               : {len(train_df_raw)}")
    print(f"Balanced train messages          : {len(train_df)}")
    print(f"Realistic test messages          : {len(test_df)}")
    print("Dataset sources                  :")
    for source, count in source_counts.items():
        print(f"  - {source}: {count}")

    trainer.train()
    metrics = trainer.evaluate()

    predictions = trainer.predict(test_dataset)
    y_true = test_df["label_num"].to_numpy()
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_prob = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    report = classification_report(
        y_true,
        y_pred,
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    prediction_rows = test_df[["source", "label", "message"]].copy()
    prediction_rows["predicted_label"] = np.where(y_pred == 1, "spam", "ham")
    prediction_rows["ham_probability"] = y_prob[:, 0]
    prediction_rows["spam_probability"] = y_prob[:, 1]
    prediction_rows["is_correct"] = prediction_rows["label"] == prediction_rows["predicted_label"]

    final_metrics = {
        "model": BASE_MODEL,
        "random_state": RANDOM_STATE,
        "max_length": MAX_LENGTH,
        "balanced_training_data": BALANCE_TRAINING_DATA,
        "dataset_rows": int(len(df)),
        "duplicates_removed": int(duplicate_count),
        "source_counts": {source: int(count) for source, count in source_counts.items()},
        "label_distribution": label_distribution(df),
        "source_label_distribution": source_label_distribution(df),
        "raw_train_rows": int(len(train_df_raw)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "raw_train_label_distribution": label_distribution(train_df_raw),
        "train_label_distribution": label_distribution(train_df),
        "test_label_distribution": label_distribution(test_df),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": report,
        "confusion_matrix": {
            "labels": ["ham", "spam"],
            "matrix": matrix.astype(int).tolist(),
            "true_ham_pred_ham": int(matrix[0, 0]),
            "true_ham_pred_spam": int(matrix[0, 1]),
            "true_spam_pred_ham": int(matrix[1, 0]),
            "true_spam_pred_spam": int(matrix[1, 1]),
        },
        "trainer_eval": {key: float(value) for key, value in metrics.items()},
    }

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    METRICS_PATH.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    prediction_rows.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")
    save_evaluation_chart(y_true, y_pred, final_metrics)

    print("\nFinal DistilBERT metrics")
    print(f"Accuracy : {final_metrics['test_accuracy']:.4f}")
    print(f"Precision: {final_metrics['test_precision']:.4f}")
    print(f"Recall   : {final_metrics['test_recall']:.4f}")
    print(f"F1-score : {final_metrics['test_f1']:.4f}")
    print(f"\nSaved model   : {MODEL_DIR}")
    print(f"Saved metrics : {METRICS_PATH}")
    print(f"Saved chart   : {CHART_PATH}")
    print(f"Saved test CSV: {PREDICTIONS_PATH}")


def save_evaluation_chart(y_true, y_pred, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("DistilBERT SMS Spam Evaluation", fontsize=14, fontweight="bold")

    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    values = [
        metrics["test_accuracy"],
        metrics["test_precision"],
        metrics["test_recall"],
        metrics["test_f1"],
    ]
    bars = axes[0].bar(metric_names, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Test Metrics")
    axes[0].set_ylabel("Score")
    for bar, value in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            fontsize=10,
        )

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Ham", "Spam"],
        cmap="Blues",
        colorbar=False,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def load_saved_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"DistilBERT model not found at {MODEL_DIR}. Run python distilbert_sms_spam.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    return tokenizer, model


def predict(message):
    signal_score, signal_matches = spam_signal_score(message)
    tokenizer, model = load_saved_model()
    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    pred = int(np.argmax(probabilities))
    label = "Spam" if pred == 1 else "Ham"
    confidence = float(probabilities[pred])
    if label == "Ham" and signal_score >= 2:
        return "Spam", max(0.85, 1 - confidence), signal_matches
    return label, confidence, signal_matches


def main():
    parser = argparse.ArgumentParser(description="Train or run DistilBERT SMS spam detection.")
    parser.add_argument("--predict", "-p", nargs="+", help="Predict one SMS message.")
    args = parser.parse_args()

    if args.predict:
        message = " ".join(args.predict)
        label, confidence, signal_matches = predict(message)
        print(f"Message    : {message}")
        print(f"Prediction : {label}")
        print(f"Confidence : {confidence:.2%}")
        if signal_matches:
            print("Spam signals found in the message.")
    else:
        train_model()


if __name__ == "__main__":
    main()
