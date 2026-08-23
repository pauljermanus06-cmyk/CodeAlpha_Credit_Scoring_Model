import os
import random
import json
import sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib
from feature_engineering import fit_transformer_on_train, transform_with_preprocessor

print("current working directory:", os.getcwd())

# Prefer loading an existing CSV provided by the user. Look for common filenames.
data_candidates = ['loan_data.csv', 'loan_data_generated.csv']
df = None
for p in data_candidates:
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            print(f"Loaded data from {p}")
            break
        except Exception as e:
            print(f"Failed to read {p}: {e}")

if df is None:
    print('No loan CSV found. Please place `loan_data.csv` in the project folder.')
    raise SystemExit(1)

print(df)
print("Rows:", len(df))

print("First up to 100 rows:")
print(df.head(100))

print("\nmissing values:")
print(df.isnull().sum())

x = df.drop(['Default', 'AccountNumber'], axis=1)
y = df['Default']

print("\nFeatures(x):")
print(x.head())

print("\nTarget(y):")
print(y.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# split then fit preprocessing on train only
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("training data: ", x_train.shape)
print("testing data:", x_test.shape)

# fit preprocessor on training fold and transform
preprocessor, X_train_trans = fit_transformer_on_train(x_train)
X_test_trans = transform_with_preprocessor(preprocessor, x_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_trans, y_train)
print("Model trained successfully.")

# evaluate on transformed test set
y_pred = model.predict(X_test_trans)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy *100:.2f}%")

# save model and preprocessor
joblib.dump(model, 'credit_scoring_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("model and preprocessor saved successfully.")

# small helper for labeled output
def label_result(value):
    return "Loan Approved(low risk)" if value == 1 else "Loan Denied(high risk)"

# Use the same feature pipeline to make predictions from the full input data
X_full_trans = transform_with_preprocessor(preprocessor, x)
prediction = model.predict(X_full_trans)
df['Prediction'] = prediction
df['Prediction_Label'] = df['Prediction'].apply(label_result)



def analyze_account(account_number, df, preprocessor, model):
    # find the record for the account number
    try:
        acct_int = int(str(account_number).replace('ACC', '').replace('acc', ''))
    except Exception:
        try:
            acct_int = int(account_number)
        except Exception:
            return None
    rec = df[df['AccountNumber'] == acct_int]
    if rec.empty:
        return None
    # prepare features (drop non-feature cols) — keep Payment History if present
    X_rec = rec.drop(['Default', 'Prediction', 'Prediction_Label', 'AccountNumber'], axis=1, errors='ignore')
    # Ensure columns expected by the preprocessor are present; fill missing with NaN
    try:
        required = list(getattr(preprocessor, 'feature_names_in_', []))
        if required:
            X_rec = X_rec.reindex(columns=required, fill_value=np.nan)
    except Exception:
        pass
    X_trans = transform_with_preprocessor(preprocessor, X_rec)
    pred = model.predict(X_trans)[0]
    credit_score = int(rec.iloc[0]['Credit Score']) if 'Credit Score' in rec.columns else None
    return {
        'AccountNumber': int(rec.iloc[0]['AccountNumber']),
        'Credit Score': credit_score,
        'Prediction': int(pred),
        'Prediction_Label': label_result(pred)
    }


def retrain_model_from_df(df, model_path='credit_scoring_model.pkl', preprocessor_path='preprocessor.pkl'):
    # retrain pipeline from dataframe and resave model + preprocessor
    X = df.drop(['Default', 'AccountNumber'], axis=1)
    y = df['Default']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor_new, X_train_trans = fit_transformer_on_train(X_train)
    X_test_trans = transform_with_preprocessor(preprocessor_new, X_test)
    new_model = LogisticRegression(max_iter=1000)
    new_model.fit(X_train_trans, y_train)
    # evaluate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_pred_test = new_model.predict(X_test_trans)
    # some classifiers may not implement predict_proba; guard it
    try:
        y_proba_test = new_model.predict_proba(X_test_trans)[:, 1]
    except Exception:
        y_proba_test = None
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    roc = roc_auc_score(y_test, y_proba_test) if y_proba_test is not None else None
    # save
    joblib.dump(new_model, model_path)
    joblib.dump(preprocessor_new, preprocessor_path)
    # update predictions in df
    X_full = X
    X_full_trans = transform_with_preprocessor(preprocessor_new, X_full)
    df['Prediction'] = new_model.predict(X_full_trans)
    df['Prediction_Label'] = df['Prediction'].apply(label_result)
    # print extended metrics
    print(f"Retrain results — Accuracy: {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}" if roc is not None else "ROC-AUC  : n/a (no predict_proba)")
    return new_model, preprocessor_new, acc


print("\nPredictions for input CSV:")
print(df[['Prediction', 'Prediction_Label']].head())


if __name__ == "__main__":
    # simple interactive CLI: query by account number and optionally provide feedback
    print('\nInteractive account lookup. Enter an account number to analyse.')
    try:
        model = joblib.load('credit_scoring_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
    except Exception:
        print('Saved model or preprocessor not found; exiting interactive mode.')
        raise SystemExit(1)

    # prefer user-provided CSV if present
    csv_path = 'loan_data.csv'
    csv_loaded = False
    interactive_df = df
    try:
        if os.path.exists(csv_path):
            csv_df = pd.read_csv(csv_path)
            # ensure AccountNumber is int
            if 'AccountNumber' in csv_df.columns:
                try:
                    csv_df['AccountNumber'] = csv_df['AccountNumber'].astype(int)
                except Exception:
                    pass
            interactive_df = csv_df
            csv_loaded = True
    except Exception:
        interactive_df = df

    try:
        while True:
            acct = input('AccountNumber (or "exit" to quit): ').strip()
            if acct.lower() in ('exit', 'quit'):
                break
            info = analyze_account(acct, interactive_df, preprocessor, model)
            if info is None:
                print('Account not found. Try another one.')
                continue
            print(f"Account: {info['AccountNumber']}\nCredit Score: {info['Credit Score']}\nResult: {info['Prediction_Label']}\n")
            fb = input('Provide feedback? (y/n): ').strip().lower()
            if fb == 'y':
                actual = input('Did the user default? (1 = defaulted/high risk, 0 = repaid/low risk): ').strip()
                try:
                    actual_int = int(actual)
                    idx = interactive_df[interactive_df['AccountNumber'] == int(acct)].index
                    if not idx.empty:
                        interactive_df.loc[idx, 'Default'] = actual_int
                        # persist the updated label to a safe copy (never overwrite original CSV)
                        try:
                            fallback = csv_path.replace('.csv', '_updated.csv')
                            interactive_df.to_csv(fallback, index=False)
                            print(f'Wrote updated labels to: {fallback}')
                        except Exception as e:
                            print(f'Failed to write updated labels to fallback file: {e}')
                        print('Retraining model with updated label...')
                        model, preprocessor, new_acc = retrain_model_from_df(interactive_df)
                        print(f'Retrained; new test accuracy: {new_acc*100:.2f}%')
                    else:
                        print('Account index not found when applying feedback.')
                except Exception as e:
                    print('Invalid feedback input, skipping retrain.', e)
    except (KeyboardInterrupt, EOFError):
        print('\nInterrupted — exiting interactive mode.')
        sys.exit(0)
