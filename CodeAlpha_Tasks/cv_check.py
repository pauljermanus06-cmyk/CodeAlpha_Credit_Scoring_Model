import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from feature_engineering import build_preprocessor

# pick csv
csv = 'loan_data.csv' if os.path.exists('loan_data.csv') else ('loan_data_generated.csv' if os.path.exists('loan_data_generated.csv') else None)
if csv is None:
    print('No CSV found (loan_data.csv or loan_data_generated.csv).')
    raise SystemExit(1)

df = pd.read_csv(csv)
if 'AccountNumber' in df.columns:
    try:
        df['AccountNumber'] = df['AccountNumber'].astype(int)
    except Exception:
        pass

print('Loaded', csv, 'rows=', len(df))
print('Class counts:\n', df['Default'].value_counts())

X = df.drop(['Default','AccountNumber'], axis=1, errors='ignore')
y = df['Default']

# build preprocessor
pre = build_preprocessor(X)
pipe = Pipeline([('preprocessor', pre), ('clf', LogisticRegression(max_iter=1000))])

scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
roc_scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
prec_scores = cross_val_score(pipe, X, y, cv=5, scoring='precision')
rec_scores = cross_val_score(pipe, X, y, cv=5, scoring='recall')
f1_scores = cross_val_score(pipe, X, y, cv=5, scoring='f1')
print('5-fold accuracy scores:', np.round(scores,3))
print('5-fold roc_auc scores:', np.round(roc_scores,3))
print('5-fold precision scores:', np.round(prec_scores,3))
print('5-fold recall scores:', np.round(rec_scores,3))
print('5-fold f1 scores:', np.round(f1_scores,3))
print('Mean accuracy: {:.3f} (std {:.3f})'.format(scores.mean(), scores.std()))

# replicate train/test split used by script
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# get indices
train_idx = X_train.index
test_idx = X_test.index

acct = 1200031008
if 'AccountNumber' in df.columns and acct in df['AccountNumber'].values:
    acct_idx = df.index[df['AccountNumber']==acct].tolist()[0]
    where = 'test' if acct_idx in test_idx else ('train' if acct_idx in train_idx else 'neither')
    print(f'Account {acct} index {acct_idx} ended up in: {where} set')
else:
    print(f'Account {acct} not present in CSV.')

# train and evaluate
pre2 = build_preprocessor(X_train)
X_train_t = pre2.fit_transform(X_train)
X_test_t = pre2.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_t, y_train)
y_pred = model.predict(X_test_t)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
try:
    y_proba = model.predict_proba(X_test_t)[:,1]
    roc = roc_auc_score(y_test, y_proba)
except Exception:
    roc = None
cm = confusion_matrix(y_test, y_pred)
print(f'Train/test eval accuracy: {acc:.3f}')
print(f'Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, ROC-AUC: {roc if roc is not None else "n/a"}')
print('Confusion matrix (test):')
print(cm)
print('\nClassification report (test):')
print(classification_report(y_test, y_pred, zero_division=0))
