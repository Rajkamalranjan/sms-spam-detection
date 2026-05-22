"""Shared dataset loading utilities for SMS spam experiments."""

import csv
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "SMSSpamCollection.txt"
EXTRA_DATA_DIR = BASE_DIR / "datasets"

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
