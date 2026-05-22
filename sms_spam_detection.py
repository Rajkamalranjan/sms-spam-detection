"""
========================================================
  SMS SPAM DETECTION - Complete ML Project
========================================================
  Dataset  : SMS Spam Collection (5574 messages)
  Task     : Binary Classification (spam vs ham)
  Models   : Naive Bayes, Logistic Regression, SVM, Random Forest
  Author   : Generated for educational/project use
========================================================
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import joblib
import re
import string
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

from collections import Counter

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

from data_utils import load_data

# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  SMS SPAM DETECTION — COMPLETE PROJECT")
print("=" * 60)

df, duplicates_removed, source_counts = load_data()

print(f"\n✅ Dataset loaded successfully")
print(f"   Total messages : {len(df)}")
print(f"   Spam messages  : {(df['label']=='spam').sum()}")
print(f"   Ham messages   : {(df['label']=='ham').sum()}")
print(f"   Duplicates removed : {duplicates_removed}")
print("   Source counts:")
for source, count in source_counts.items():
    print(f"     - {source}: {count}")
print(f"\nSample data:")
print(df.head(5).to_string(index=False))

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n✅ Output directory created or exists: {OUT_DIR}")

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  EXPLORATORY DATA ANALYSIS")
print("─" * 60)

# Feature Engineering
df["label_num"]     = df["label"].map({"ham": 0, "spam": 1})
df["msg_length"]    = df["message"].apply(len)
df["word_count"]    = df["message"].apply(lambda x: len(x.split()))
df["char_count"]    = df["message"].apply(lambda x: len(x.replace(" ", "")))
df["digit_count"]   = df["message"].apply(lambda x: sum(c.isdigit() for c in x))
df["upper_count"]   = df["message"].apply(lambda x: sum(c.isupper() for c in x))
df["punct_count"]   = df["message"].apply(lambda x: sum(c in string.punctuation for c in x))
df["has_url"]       = df["message"].str.contains(r"http|www|\.com", case=False).astype(int)
df["has_phone"]     = df["message"].str.contains(r"\d{5,}", case=False).astype(int)
df["has_currency"]  = df["message"].str.contains(r"£|\$|€|free|win|prize|cash", case=False).astype(int)
df["exclaim_count"] = df["message"].apply(lambda x: x.count("!"))

print("\nStatistics by label:")
print(df.groupby("label")[["msg_length","word_count","digit_count","upper_count"]].describe().round(2))

# ─────────────────────────────────────────────
# 4. VISUALIZATIONS — Figure 1: EDA
# ─────────────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle("SMS Spam Detection — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=0.98)

colors = {"ham": "#2ecc71", "spam": "#e74c3c"}

# 4a. Class distribution (donut)
ax = axes[0, 0]
counts = df["label"].value_counts()
wedges, texts, autotexts = ax.pie(
    counts, labels=counts.index, autopct="%1.1f%%",
    colors=[colors["ham"], colors["spam"]],
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2)
)
for t in autotexts: t.set_fontsize(12); t.set_fontweight("bold")
ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
ax.text(0, 0, f"n={len(df)}", ha="center", va="center", fontsize=12, fontweight="bold")

# 4b. Message length distribution
ax = axes[0, 1]
for label, grp in df.groupby("label"):
    ax.hist(grp["msg_length"], bins=40, alpha=0.7, label=label,
            color=colors[label], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Message Length (chars)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Message Length Distribution", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.axvline(df[df.label=="spam"]["msg_length"].median(), color="#e74c3c", linestyle="--", alpha=0.8)
ax.axvline(df[df.label=="ham"]["msg_length"].median(),  color="#2ecc71", linestyle="--", alpha=0.8)

# 4c. Word count boxplot
ax = axes[0, 2]
spam_wc = df[df.label=="spam"]["word_count"]
ham_wc  = df[df.label=="ham"]["word_count"]
bp = ax.boxplot([ham_wc, spam_wc], labels=["Ham", "Spam"],
                patch_artist=True, notch=True,
                medianprops=dict(color="white", linewidth=2))
bp["boxes"][0].set_facecolor("#2ecc71")
bp["boxes"][1].set_facecolor("#e74c3c")
ax.set_ylabel("Word Count", fontsize=11)
ax.set_title("Word Count by Class", fontsize=13, fontweight="bold")

# 4d. Feature comparison heatmap
ax = axes[1, 0]
feat_cols = ["msg_length","word_count","digit_count","upper_count","punct_count","has_url","has_phone","has_currency","exclaim_count"]
feat_means = df.groupby("label")[feat_cols].mean().T
feat_norm  = feat_means.div(feat_means.max(axis=1), axis=0)
im = ax.imshow(feat_norm.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Ham", "Spam"], fontsize=11)
ax.set_yticks(range(len(feat_cols))); ax.set_yticklabels(feat_cols, fontsize=9)
for i in range(feat_norm.shape[0]):
    for j in range(feat_norm.shape[1]):
        ax.text(j, i, f"{feat_norm.values[i,j]:.2f}", ha="center", va="center",
                fontsize=8, color="black", fontweight="bold")
ax.set_title("Feature Comparison\n(Normalized)", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 4e. Spam indicator features
ax = axes[1, 1]
indicator_feats = ["has_url","has_phone","has_currency"]
spam_rates = {}
for feat in indicator_feats:
    pct = df.groupby(feat)["label_num"].mean() * 100
    spam_rates[feat] = pct.get(1, 0)
bars = ax.bar(indicator_feats, [spam_rates[f] for f in indicator_feats],
              color=["#3498db","#9b59b6","#e67e22"], edgecolor="white", linewidth=1.5)
ax.set_ylabel("% Spam", fontsize=11)
ax.set_title("Spam Rate by Indicator Feature", fontsize=13, fontweight="bold")
ax.set_ylim(0, 100)
for bar, feat in zip(bars, indicator_feats):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{spam_rates[feat]:.1f}%", ha="center", fontsize=11, fontweight="bold")

# 4f. Uppercase usage
ax = axes[1, 2]
df["upper_pct"] = df["upper_count"] / df["msg_length"] * 100
for label, grp in df.groupby("label"):
    data = grp["upper_pct"].clip(0, 50)
    ax.hist(data, bins=30, alpha=0.7, label=label, color=colors[label],
            edgecolor="white", linewidth=0.5, density=True)
ax.set_xlabel("% Uppercase Characters", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Uppercase Usage by Class", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(OUT_DIR / "01_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ EDA chart saved → 01_eda.png")


# ─────────────────────────────────────────────
# 5. TEXT PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  TEXT PREPROCESSING")
print("─" * 60)

STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
    "won","wouldn"
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " urltoken ", text)       # URLs
    text = re.sub(r"\b\d[\d\s]{3,}\d\b", " phonetoken ", text) # Phone-like numbers
    text = re.sub(r"£|\$|€", " currencytoken ", text)           # Currency symbols
    text = re.sub(r"[^a-z\s]", " ", text)                       # Remove non-alpha
    text = re.sub(r"\s+", " ", text).strip()                    # Collapse spaces
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)

df["clean_message"] = df["message"].apply(preprocess)

print("\nPreprocessing example:")
sample = df[df.label=="spam"].iloc[2]
print(f"  Original : {sample['message'][:100]}")
print(f"  Cleaned  : {sample['clean_message'][:100]}")

# Top words per class
def top_words(series, n=15):
    words = " ".join(series).split()
    return Counter(words).most_common(n)

spam_top = top_words(df[df.label=="spam"]["clean_message"])
ham_top  = top_words(df[df.label=="ham"]["clean_message"])

# ─────────────────────────────────────────────
# 6. VISUALIZATION — Figure 2: Top Words
# ─────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle("Most Frequent Words by Class", fontsize=15, fontweight="bold")

for ax, top, label, color in [
    (axes[0], ham_top,  "Ham (Legitimate)",  "#2ecc71"),
    (axes[1], spam_top, "Spam",              "#e74c3c"),
]:
    words, counts = zip(*top)
    bars = ax.barh(list(words)[::-1], list(counts)[::-1], color=color, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_title(label, fontsize=13, fontweight="bold", color=color)
    for bar, count in zip(bars, list(counts)[::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                str(count), va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "02_top_words.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Top words chart saved → 02_top_words.png")


# ─────────────────────────────────────────────
# 7. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X = df["clean_message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n{'─'*60}")
print(f"  TRAIN / TEST SPLIT  (80/20, stratified)")
print(f"{'─'*60}")
print(f"  Train : {len(X_train)} messages  (spam={y_train.sum()}, ham={len(y_train)-y_train.sum()})")
print(f"  Test  : {len(X_test)} messages   (spam={y_test.sum()}, ham={len(y_test)-y_test.sum()})")


# ─────────────────────────────────────────────
# 8. MODEL TRAINING
# ─────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  MODEL TRAINING")
print(f"{'─'*60}")

VECTORIZER = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)

models = {
    "Naive Bayes (MNB)"       : Pipeline([("tfidf", VECTORIZER), ("clf", MultinomialNB(alpha=0.1))]),
    "Logistic Regression"     : Pipeline([("tfidf", VECTORIZER), ("clf", LogisticRegression(C=10, max_iter=1000, random_state=42))]),
    "Linear SVM"              : Pipeline([("tfidf", VECTORIZER), ("clf", LinearSVC(C=1.0, random_state=42, dual=False))]),
    "Random Forest"           : Pipeline([("tfidf", VECTORIZER), ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]),
    "Gradient Boosting"       : Pipeline([("tfidf", VECTORIZER), ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))]),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n  Training: {name} ...", end=" ")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ROC AUC — LinearSVC needs decision_function
    if hasattr(model.named_steps["clf"], "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    results[name] = {
        "model"    : model,
        "y_pred"   : y_pred,
        "y_score"  : y_score,
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "f1"       : f1_score(y_test, y_pred),
        "roc_auc"  : roc_auc_score(y_test, y_score),
        "cv_f1"    : cv_scores.mean(),
        "cv_std"   : cv_scores.std(),
    }
    print(f"Accuracy={results[name]['accuracy']:.4f}  F1={results[name]['f1']:.4f}  ROC-AUC={results[name]['roc_auc']:.4f}")

# ─────────────────────────────────────────────
# 9. RESULTS TABLE
# ─────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  MODEL COMPARISON RESULTS")
print(f"{'─'*60}")

summary = pd.DataFrame({
    name: {
        "Accuracy"  : f"{r['accuracy']:.4f}",
        "Precision" : f"{r['precision']:.4f}",
        "Recall"    : f"{r['recall']:.4f}",
        "F1-Score"  : f"{r['f1']:.4f}",
        "ROC-AUC"   : f"{r['roc_auc']:.4f}",
        "CV F1 (5k)": f"{r['cv_f1']:.4f} ± {r['cv_std']:.4f}",
    }
    for name, r in results.items()
}).T

print(summary.to_string())

best_model_name = max(results, key=lambda n: results[n]["f1"])
print(f"\n🏆 Best Model: {best_model_name}  (F1={results[best_model_name]['f1']:.4f})")


# ─────────────────────────────────────────────
# 10. VISUALIZATION — Figure 3: Model Comparison
# ─────────────────────────────────────────────
fig3, axes = plt.subplots(2, 3, figsize=(18, 12))
fig3.suptitle("Model Comparison & Evaluation", fontsize=16, fontweight="bold", y=0.98)

model_names_short = [n.replace(" (MNB)","") for n in results]
metrics = ["accuracy","precision","recall","f1","roc_auc"]
metric_labels = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
palette = ["#3498db","#2ecc71","#e74c3c","#9b59b6","#f39c12"]

# 10a. Metric bars
ax = axes[0, 0]
x = np.arange(len(metrics))
w = 0.14
for i, (name, r) in enumerate(results.items()):
    vals = [r[m] for m in metrics]
    bars = ax.bar(x + i*w, vals, w, label=name.replace(" (MNB)",""), color=palette[i], alpha=0.85, edgecolor="white")

ax.set_xticks(x + w*2)
ax.set_xticklabels(metric_labels, fontsize=10, rotation=15)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0.85, 1.01)
ax.set_title("All Metrics Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=0.3)

# 10b. ROC Curves
ax = axes[0, 1]
for i, (name, r) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, r["y_score"])
    ax.plot(fpr, tpr, color=palette[i], lw=2,
            label=f"{name.replace(' (MNB)','').replace('Logistic Regression','LR')} (AUC={r['roc_auc']:.3f})")
ax.plot([0,1],[0,1], "k--", alpha=0.4, lw=1)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

# 10c–10g. Confusion matrices (best 3 + worst 1 shown as 4 subplots)
selected = [best_model_name, "Naive Bayes (MNB)", "Random Forest", "Gradient Boosting"]
for idx, (name, ax_) in enumerate(zip(selected, [axes[0,2], axes[1,0], axes[1,1], axes[1,2]])):
    r = results[name]
    cm = confusion_matrix(y_test, r["y_pred"])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_,
                xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"],
                linewidths=1, linecolor="white", cbar=False)
    for i in range(2):
        for j in range(2):
            ax_.text(j+0.5, i+0.75, f"({cm_pct[i,j]:.1f}%)",
                     ha="center", va="center", fontsize=9, color="gray")
    short = name.replace(" (MNB)","")
    ax_.set_title(f"{short}\nF1={r['f1']:.4f}", fontsize=11, fontweight="bold")
    ax_.set_xlabel("Predicted", fontsize=10)
    ax_.set_ylabel("Actual", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "03_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Model comparison chart saved → 03_model_comparison.png")


# ─────────────────────────────────────────────
# 11. BEST MODEL — Detailed Report
# ─────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  BEST MODEL CLASSIFICATION REPORT: {best_model_name}")
print(f"{'─'*60}")
best = results[best_model_name]
print(classification_report(y_test, best["y_pred"], target_names=["Ham","Spam"]))


# ─────────────────────────────────────────────
# 12. VISUALIZATION — Figure 4: TF-IDF Feature Importance
# ─────────────────────────────────────────────
lr_model = results["Logistic Regression"]["model"]
tfidf_lr  = lr_model.named_steps["tfidf"]
clf_lr    = lr_model.named_steps["clf"]
feat_names = np.array(tfidf_lr.get_feature_names_out())
coefs      = clf_lr.coef_[0]

top_spam_idx = np.argsort(coefs)[-20:][::-1]
top_ham_idx  = np.argsort(coefs)[:20]

fig4, axes = plt.subplots(1, 2, figsize=(16, 7))
fig4.suptitle("Top TF-IDF Features — Logistic Regression Coefficients", fontsize=14, fontweight="bold")

for ax, idx, title, color in [
    (axes[0], top_spam_idx, "Top SPAM Indicators", "#e74c3c"),
    (axes[1], top_ham_idx,  "Top HAM Indicators",  "#2ecc71"),
]:
    words  = feat_names[idx]
    values = np.abs(coefs[idx])
    bars = ax.barh(words[::-1], values[::-1], color=color, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("|Coefficient|", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", color=color)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "04_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Feature importance chart saved → 04_feature_importance.png")


# ─────────────────────────────────────────────
# 13. REAL-TIME PREDICTION FUNCTION
# ─────────────────────────────────────────────
best_pipeline = results[best_model_name]["model"]
MODEL_PATH = OUT_DIR / "best_sms_spam_model.joblib"
joblib.dump(best_pipeline, MODEL_PATH)
print(f"\n✅ Best model pipeline saved → {MODEL_PATH}")

def predict_sms(message: str, model=best_pipeline):
    """Predict whether a single SMS is spam or ham."""
    clean = preprocess(message)
    pred  = model.predict([clean])[0]
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba([clean])[0]
        confidence = proba[pred]
    else:
        score = model.decision_function([clean])[0]
        confidence = 1 / (1 + np.exp(-abs(score)))
    label = "SPAM 🚨" if pred == 1 else "HAM  ✅"
    return label, confidence

print(f"\n{'─'*60}")
print("  LIVE PREDICTIONS (Sample Messages)")
print(f"{'─'*60}")

test_messages = [
    "WINNER!! You have been selected to receive a £900 prize. Call 09061701461 now!",
    "Hey, are you coming to the party tonight? Let me know!",
    "FREE entry! Text WIN to 84484 now! TCs apply, std msg charge",
    "I'll be home late today. Can you start dinner without me?",
    "Congratulations! You've won a brand new iPhone. Click here to claim.",
    "Can we reschedule our meeting to 3pm tomorrow?",
    "Urgent! Your bank account has been suspended. Verify at www.fake-bank.com",
]

for msg in test_messages:
    label, conf = predict_sms(msg)
    print(f"\n  Message    : {msg[:70]}{'...' if len(msg)>70 else ''}")
    print(f"  Prediction : {label}  (Confidence: {conf:.2%})")

parser = argparse.ArgumentParser(description="SMS Spam Detection — real-time prediction interface")
parser.add_argument("--predict", "-p", nargs="+", help="Predict one or more SMS messages. Wrap multi-word text in quotes.")
args = parser.parse_args()
if args.predict:
    user_message = " ".join(args.predict)
    label, conf = predict_sms(user_message)
    print(f"\n{'─'*60}")
    print("  USER INPUT PREDICTION")
    print(f"{'─'*60}")
    print(f"  Message    : {user_message}")
    print(f"  Prediction : {label}  (Confidence: {conf:.2%})")

# ─────────────────────────────────────────────
# 14. SUMMARY REPORT
# ─────────────────────────────────────────────
print(f"\n{'═'*60}")
print("  PROJECT SUMMARY")
print(f"{'═'*60}")
print(f"  Dataset     : {len(df):,} messages ({(df['label']=='spam').sum():,} spam, {(df['label']=='ham').sum():,} ham)")
print(f"  Features    : TF-IDF (unigrams + bigrams, 8000 features)")
print(f"  Train/Test  : 80% / 20% stratified split")
print(f"\n  {'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print(f"  {'─'*25} {'─'*9} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
for name, r in results.items():
    star = " ⭐" if name == best_model_name else ""
    short = name.replace(" (MNB)","(NB)")[:25]
    print(f"  {short:<25} {r['accuracy']:>9.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>8.4f}{star}")
print(f"\n  Best model  : {best_model_name}")
print(f"  F1-Score    : {results[best_model_name]['f1']:.4f}")
print(f"  ROC-AUC     : {results[best_model_name]['roc_auc']:.4f}")
print(f"\n  Output files:")
print(f"    01_eda.png                — Exploratory Data Analysis")
print(f"    02_top_words.png          — Top words per class")
print(f"    03_model_comparison.png   — Model metrics & confusion matrices")
print(f"    04_feature_importance.png — LR TF-IDF feature weights")
print(f"{'═'*60}")
print("  ✅ Project Complete!")
print(f"{'═'*60}")
