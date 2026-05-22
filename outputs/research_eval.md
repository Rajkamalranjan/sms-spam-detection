# Research Evaluation

Model: `F:\New folder\outputs\best_sms_spam_model.joblib`
Dataset rows: 18202
Test rows: 3641
Duplicates removed: 414

## Overall Test Metrics
rows=3641, accuracy=0.9959, precision=0.9986, recall=0.9909, F1=0.9947, ROC-AUC=0.9998

## Real SMS Only Metrics
rows=1028, accuracy=0.9883, precision=0.9848, recall=0.9286, F1=0.9559, ROC-AUC=0.9958

## Source-wise Metrics

Sources with fewer than 30 test rows should be treated as qualitative checks, not stable estimates.
- sms_spam_collection: rows=1028, accuracy=0.9883, precision=0.9848, recall=0.9286, F1=0.9559, ROC-AUC=0.9958
- smsdata: rows=2603, accuracy=1.0000, precision=1.0000, recall=1.0000, F1=1.0000, ROC-AUC=1.0000
- synthetic_chat: rows=2, accuracy=0.5000, precision=1.0000, recall=0.5000, F1=0.6667
- synthetic_email: rows=4, accuracy=0.5000, precision=1.0000, recall=0.3333, F1=0.5000, ROC-AUC=1.0000
- synthetic_social: rows=4, accuracy=1.0000, precision=1.0000, recall=1.0000, F1=1.0000, ROC-AUC=1.0000
