"""
Create research-friendly dataset audit artifacts for the spam detection project.

Run:
    python dataset_audit.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from data_utils import load_data


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
AUDIT_JSON = OUT_DIR / "dataset_audit.json"
AUDIT_MD = OUT_DIR / "dataset_audit.md"
AUDIT_CHART = OUT_DIR / "06_dataset_audit.png"


def count_pattern(series, pattern):
    return int(series.str.contains(pattern, case=False, regex=True).sum())


def build_audit():
    df, duplicates_removed, source_counts = load_data()
    text = df["message"]
    label_counts = df["label"].value_counts().sort_index()
    by_source_label = df.groupby(["source", "label"]).size().unstack(fill_value=0).sort_index()

    return {
        "total_rows": int(len(df)),
        "duplicates_removed": int(duplicates_removed),
        "label_counts": {label: int(count) for label, count in label_counts.items()},
        "spam_ratio": float((df["label"] == "spam").mean()),
        "source_counts": {source: int(count) for source, count in source_counts.items()},
        "source_label_counts": {
            source: {label: int(count) for label, count in row.items()}
            for source, row in by_source_label.iterrows()
        },
        "message_length": {
            "min": int(text.str.len().min()),
            "median": float(text.str.len().median()),
            "mean": float(text.str.len().mean()),
            "max": int(text.str.len().max()),
        },
        "word_count": {
            "min": int(text.str.split().str.len().min()),
            "median": float(text.str.split().str.len().median()),
            "mean": float(text.str.split().str.len().mean()),
            "max": int(text.str.split().str.len().max()),
        },
        "signal_counts": {
            "contains_url": count_pattern(text, r"http\S+|www\S+"),
            "contains_phone_like_number": count_pattern(text, r"\b\d{5,}\b"),
            "contains_money_or_reward_terms": count_pattern(
                text,
                r"\$|rs\s?\d+|free|won|winner|prize|cash|reward|claim",
            ),
            "contains_urgency_terms": count_pattern(
                text,
                r"urgent|now|today|final|immediately|expires|blocked|suspended",
            ),
            "contains_otp_or_kyc": count_pattern(text, r"\botp\b|\bkyc\b"),
        },
    }


def save_chart(audit):
    labels = list(audit["label_counts"].keys())
    values = [audit["label_counts"][label] for label in labels]
    source_names = list(audit["source_counts"].keys())
    source_values = [audit["source_counts"][source] for source in source_names]
    signal_names = list(audit["signal_counts"].keys())
    signal_values = [audit["signal_counts"][signal] for signal in signal_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Dataset Audit for SMS Spam Detection", fontsize=14, fontweight="bold")

    axes[0].bar(labels, values, color=["#4C78A8", "#E45756"])
    axes[0].set_title("Class Balance")
    axes[0].set_ylabel("Rows")

    axes[1].barh(source_names, source_values, color="#72B7B2")
    axes[1].set_title("Rows by Source")
    axes[1].set_xlabel("Rows")

    axes[2].barh([name.replace("_", " ") for name in signal_names], signal_values, color="#F58518")
    axes[2].set_title("Spam-Signal Coverage")
    axes[2].set_xlabel("Rows")

    plt.tight_layout()
    plt.savefig(AUDIT_CHART, dpi=150, bbox_inches="tight")
    plt.close()


def save_markdown(audit):
    lines = [
        "# Dataset Audit",
        "",
        f"Total rows after duplicate removal: {audit['total_rows']}",
        f"Duplicates removed: {audit['duplicates_removed']}",
        f"Spam ratio: {audit['spam_ratio']:.3f}",
        "",
        "## Label Counts",
    ]
    for label, count in audit["label_counts"].items():
        lines.append(f"- {label}: {count}")

    lines.extend(["", "## Source Counts"])
    for source, count in audit["source_counts"].items():
        lines.append(f"- {source}: {count}")

    lines.extend(["", "## Text Statistics"])
    for section in ("message_length", "word_count"):
        stats = audit[section]
        readable = section.replace("_", " ").title()
        lines.append(
            f"- {readable}: min={stats['min']}, median={stats['median']:.1f}, "
            f"mean={stats['mean']:.1f}, max={stats['max']}"
        )

    lines.extend(["", "## Signal Counts"])
    for signal, count in audit["signal_counts"].items():
        lines.append(f"- {signal}: {count}")

    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    save_markdown(audit)
    save_chart(audit)
    print(f"Saved audit JSON : {AUDIT_JSON}")
    print(f"Saved audit MD   : {AUDIT_MD}")
    print(f"Saved audit chart: {AUDIT_CHART}")


if __name__ == "__main__":
    main()
