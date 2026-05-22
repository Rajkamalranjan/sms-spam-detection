# SMS Spam Detection Research Project

This project studies binary SMS/message spam detection using two modeling tracks:

- classical machine learning with TF-IDF features and sklearn classifiers
- transformer fine-tuning with DistilBERT

The repository includes the original SMS Spam Collection data, a generated multi-channel dataset, dataset audit artifacts, model comparison charts, and a Streamlit prediction interface.

## Project Structure

```text
.
├── SMSSpamCollection.txt              # Original SMS Spam Collection dataset
├── datasets/
│   └── smsdata.csv                    # Generated multi-channel ham/spam data
├── sms_spam_detection.py              # Classical ML training and charts
├── distilbert_sms_spam.py             # DistilBERT training and prediction
├── data_utils.py                      # Shared dataset loading and normalization
├── research_eval.py                   # Research-grade source-wise evaluation
├── dataset_audit.py                   # Dataset audit JSON, Markdown, and chart
├── generate_extra_spam_dataset.py     # Synthetic data generator
├── streamlit_app.py                   # Interactive demo app
├── requirements.txt
└── outputs/                           # Generated reports, charts, and models
```

## Research Goal

The goal is to compare lightweight TF-IDF models against a fine-tuned transformer for detecting spam messages while documenting data provenance, class balance, synthetic-data impact, and source-wise generalization.

This matters because a model can score very highly on a mixed dataset while still underperforming on real-world SMS messages if most training examples are generated from templates.

## Dataset Summary

The current audited dataset contains 18,202 rows after duplicate removal:

- ham: 11,038
- spam: 7,164
- spam ratio: 0.394
- original SMS source: 5,158 rows
- generated `smsdata` source: 13,000 rows
- built-in synthetic examples: 44 rows

Regenerate this summary with:

```bash
python dataset_audit.py
```

## Reproducible Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the extra dataset:

```bash
python generate_extra_spam_dataset.py
```

Audit the full dataset:

```bash
python dataset_audit.py
```

Train classical ML models:

```bash
python sms_spam_detection.py
```

Train DistilBERT:

```bash
python distilbert_sms_spam.py
```

Run research evaluation:

```bash
python research_eval.py
```

Launch the app:

```bash
streamlit run streamlit_app.py
```

## Evaluation Protocol

For research-level reporting, use more than one metric view:

- overall accuracy, precision, recall, F1, and ROC-AUC
- confusion matrix
- source-wise performance
- real-SMS-only performance on `sms_spam_collection`
- synthetic-source performance on generated rows

The `research_eval.py` script evaluates the saved sklearn pipeline and writes:

- `outputs/research_eval.json`
- `outputs/research_eval.md`

## Known Limitations

- The generated dataset is template-based, so it may inflate test performance if train/test splits contain similar templates.
- The saved DistilBERT model may be stale if the dataset changes after training; retrain it before using it as the primary research result.
- The Streamlit app includes simple rule-based spam-signal overrides, so app predictions are not purely neural or purely sklearn model output.
- Large model binaries are excluded from normal Git workflows. Use Git LFS or a model registry if the trained transformer must be shared.

## Recommended Reporting

For a paper, thesis, or research submission, report:

- dataset composition by source and label
- duplicate-removal policy
- preprocessing steps
- model hyperparameters
- train/test split strategy
- source-wise and real-only metrics
- limitations caused by generated data
- reproducibility details such as package versions and random seed
