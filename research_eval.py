"""
Research-oriented evaluation for the saved sklearn SMS spam model.

This script reports overall, real-only, and source-wise metrics so synthetic
data does not hide generalization problems.

Run:
    python research_eval.py
"""

import json
import re
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data_utils import load_data


RANDOM_STATE = 42


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUT_DIR / "best_sms_spam_model.joblib"
REPORT_JSON = OUT_DIR / "research_eval.json"
REPORT_MD = OUT_DIR / "research_eval.md"

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
    "shouldn", "wasn", "weren", "won", "wouldn",
}


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " urltoken ", text)
    text = re.sub(r"\b\d[\d\s]{3,}\d\b", " phonetoken ", text)
    text = re.sub(r"\u00a3|\$|\u20ac", " currencytoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in STOP_WORDS and len(word) > 1]
    return " ".join(tokens)


def score_model(model, messages):
    if hasattr(model.named_steps["clf"], "predict_proba"):
        return model.predict_proba(messages)[:, 1]
    return model.decision_function(messages)


def metric_block(y_true, y_pred, y_score=None):
    metrics = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["ham", "spam"],
            output_dict=True,
            zero_division=0,
        ),
    }
    if y_score is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def evaluate_subset(model, frame):
    clean_messages = frame["message"].map(preprocess)
    y_true = frame["label_num"].to_numpy()
    y_pred = model.predict(clean_messages)
    y_score = score_model(model, clean_messages)
    return metric_block(y_true, y_pred, y_score)


def build_report():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved sklearn model not found at {MODEL_PATH}. Run python sms_spam_detection.py first."
        )

    model = joblib.load(MODEL_PATH)
    df, duplicates_removed, source_counts = load_data()

    _, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label_num"],
    )

    report = {
        "model_path": str(MODEL_PATH),
        "random_state": RANDOM_STATE,
        "dataset_rows": int(len(df)),
        "duplicates_removed": int(duplicates_removed),
        "source_counts": {source: int(count) for source, count in source_counts.items()},
        "test_rows": int(len(test_df)),
        "overall": evaluate_subset(model, test_df),
        "real_sms_only": evaluate_subset(
            model,
            test_df[test_df["source"] == "sms_spam_collection"].copy(),
        ),
        "source_wise": {},
    }

    for source, source_df in test_df.groupby("source"):
        report["source_wise"][source] = evaluate_subset(model, source_df.copy())

    return report


def save_markdown(report):
    lines = [
        "# Research Evaluation",
        "",
        f"Model: `{report['model_path']}`",
        f"Dataset rows: {report['dataset_rows']}",
        f"Test rows: {report['test_rows']}",
        f"Duplicates removed: {report['duplicates_removed']}",
        "",
        "## Overall Test Metrics",
        metric_line(report["overall"]),
        "",
        "## Real SMS Only Metrics",
        metric_line(report["real_sms_only"]),
        "",
        "## Source-wise Metrics",
        "",
        "Sources with fewer than 30 test rows should be treated as qualitative checks, not stable estimates.",
    ]

    for source, metrics in report["source_wise"].items():
        lines.append(f"- {source}: {metric_line(metrics)}")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_line(metrics):
    auc = metrics.get("roc_auc")
    auc_text = f", ROC-AUC={auc:.4f}" if auc is not None else ""
    return (
        f"rows={metrics['rows']}, accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}{auc_text}"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_markdown(report)
    print(f"Saved research JSON: {REPORT_JSON}")
    print(f"Saved research MD  : {REPORT_MD}")
    print(metric_line(report["overall"]))


if __name__ == "__main__":
    main()
