"""
Generate an extra 12k+ multi-channel spam/ham dataset for the DistilBERT trainer.

Run:
    python generate_extra_spam_dataset.py
"""

import csv
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "datasets" / "smsdata.csv"
RANDOM_STATE = 42


SPAM_TEMPLATES = [
    "{urgency}: {account} will be {bad_action}. {cta} at {link}",
    "{brand} alert: unusual activity detected. {cta} to secure your {account}.",
    "Congratulations {name}, you won {prize}. {cta} before {deadline}.",
    "{delivery} is pending. Pay {fee} redelivery fee at {link}",
    "{job_offer}: earn {money} per day from home. Message {keyword} to start.",
    "Your {account} KYC is incomplete. Send OTP to avoid {bad_action}.",
    "{crypto} bonus active for {deadline}. Deposit {money} and double returns.",
    "Final notice: {account} access expires {deadline}. Verify now at {link}",
    "{brand} reward unlocked. Claim {prize} using code {code}.",
    "Private {content_type} found with your name. Open {link} to remove it.",
    "You are selected for {loan_amount} instant loan. Reply {keyword} now.",
    "{social} verification available today. Share login code to continue.",
]

HAM_TEMPLATES = [
    "{name}, meeting moved to {time}. Please confirm when free.",
    "Your {delivery} order is out for delivery. Track it in the official app.",
    "Reminder: {appointment} appointment is scheduled for {time}.",
    "Can you review the {document} before tomorrow's call?",
    "I transferred {money} for {reason}. Please check and confirm.",
    "The {project} update is ready. Feedback welcome when you get time.",
    "Lunch at {time}? I can book the table if that works.",
    "Your {account} statement is available in the official app.",
    "Thanks for sharing the {document}. I will read it tonight.",
    "Please bring {grocery_item} on the way back.",
    "Class notes for {subject} are uploaded in the group folder.",
    "Cab is waiting near gate {gate}. Call me if you cannot find it.",
]

VALUES = {
    "account": [
        "bank account",
        "wallet",
        "email account",
        "profile",
        "card",
        "UPI account",
        "cloud storage",
        "trading account",
    ],
    "appointment": ["doctor", "dentist", "service", "interview", "college", "support"],
    "bad_action": ["blocked", "suspended", "closed", "restricted", "disabled", "locked"],
    "brand": ["BankSecure", "PayFast", "MailBox", "ShopKart", "InstaHelp", "QuickPay"],
    "code": ["SAVE50", "WIN24", "FREE100", "VIP999", "LUCKY7", "CLAIM88"],
    "content_type": ["video", "photo", "message", "document", "post"],
    "crypto": ["crypto", "bitcoin", "forex", "trading", "investment"],
    "cta": ["click now", "verify immediately", "claim now", "update details", "confirm identity"],
    "deadline": ["today", "in 10 minutes", "before midnight", "within 24 hours", "now"],
    "delivery": ["package", "courier", "parcel", "shipment", "grocery"],
    "document": ["report", "invoice", "proposal", "notes", "presentation", "resume"],
    "fee": ["Rs 19", "Rs 29", "$1", "$2", "a small"],
    "gate": ["1", "2", "3", "4", "5"],
    "grocery_item": ["milk", "bread", "rice", "tea", "vegetables", "medicine"],
    "job_offer": ["Part-time job", "Online work", "Remote task", "Data entry work", "Video liking job"],
    "keyword": ["YES", "START", "JOIN", "OK", "WIN"],
    "link": [
        "http://secure-login.example/verify",
        "www.claim-reward.example",
        "http://tiny.example/pay",
        "www.account-check.example",
        "http://bonus.example/start",
    ],
    "loan_amount": ["Rs 50,000", "Rs 1 lakh", "$1000", "$5000", "instant cash"],
    "money": ["Rs 500", "Rs 2000", "Rs 5000", "$100", "$700"],
    "name": ["Raj", "Amit", "Priya", "Neha", "Sam", "Alex", "Customer", "Friend"],
    "prize": ["cash prize", "gift card", "free recharge", "shopping voucher", "iPhone", "bonus"],
    "project": ["dashboard", "model", "assignment", "website", "release", "dataset"],
    "reason": ["rent", "fees", "tickets", "groceries", "booking", "repair"],
    "room": ["room 101", "lab 2", "main office", "reception", "front desk", "conference room"],
    "social": ["Twitter", "Instagram", "Facebook", "LinkedIn", "Telegram"],
    "subject": ["math", "science", "ML", "history", "English", "Python"],
    "time": ["9 AM", "10:30 AM", "1 PM", "3 PM", "6:30 PM", "tomorrow"],
    "urgency": ["URGENT", "Important", "Action required", "Warning", "Final alert"],
}

CHANNELS = ["sms", "email", "twitter_dm", "whatsapp", "telegram", "promo_push"]
SPAM_EXTRAS = [
    "Do not ignore this message.",
    "Offer id {code}.",
    "Only selected users can access this.",
    "No documents required.",
    "Limited seats available.",
    "This link expires {deadline}.",
    "Reply {keyword} for details.",
    "Processing takes only 2 minutes.",
    "Your reward is already reserved.",
    "Support ticket #{ticket} is open.",
]
HAM_EXTRAS = [
    "No rush.",
    "Call me after {time}.",
    "I will be near {room}.",
    "Shared in the official group.",
    "Please confirm once done.",
    "We can discuss this later.",
    "Bring your ID if needed.",
    "I added it to the calendar.",
    "Thanks again.",
    "Reference #{ticket}.",
]


def fill_template(template, rng):
    values = {key: rng.choice(options) for key, options in VALUES.items()}
    values["ticket"] = rng.randint(1000, 9999)
    return template.format(**values)


def add_natural_variation(message, label, rng):
    extras = SPAM_EXTRAS if label == "spam" else HAM_EXTRAS
    extra = fill_template(rng.choice(extras), rng)
    if rng.random() < 0.35:
        return f"{message} {extra}"
    if rng.random() < 0.50:
        return f"{extra} {message}"
    return f"{message} {extra}"


def channelize(message, channel, label, rng):
    message = add_natural_variation(message, label, rng)
    if channel == "email":
        prefix = rng.choice(["Subject: ", "Re: ", "Alert: "]) if label == "spam" else rng.choice(["Subject: ", "Update: ", ""])
        return f"{prefix}{message}"
    if channel == "twitter_dm":
        return f"DM: {message}"
    if channel == "whatsapp":
        return f"WhatsApp: {message}"
    if channel == "telegram":
        return f"Telegram: {message}"
    if channel == "promo_push":
        return f"Notification: {message}"
    return message


def generate_rows(target_per_label=6500):
    rng = random.Random(RANDOM_STATE)
    rows = []
    seen = set()

    for label, templates in (("spam", SPAM_TEMPLATES), ("ham", HAM_TEMPLATES)):
        attempts = 0
        while sum(1 for row in rows if row["label"] == label) < target_per_label:
            attempts += 1
            if attempts > target_per_label * 20:
                raise RuntimeError(f"Could not generate enough unique {label} rows.")

            template = rng.choice(templates)
            channel = rng.choice(CHANNELS)
            text = channelize(fill_template(template, rng), channel, label, rng)
            key = (label, text.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"label": label, "text": text})

    rng.shuffle(rows)
    return rows


def main():
    rows = generate_rows()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "text"])
        writer.writeheader()
        writer.writerows(rows)

    spam_count = sum(1 for row in rows if row["label"] == "spam")
    ham_count = len(rows) - spam_count
    print(f"Saved {len(rows)} rows to {OUT_PATH}")
    print(f"Spam: {spam_count}")
    print(f"Ham : {ham_count}")


if __name__ == "__main__":
    main()
