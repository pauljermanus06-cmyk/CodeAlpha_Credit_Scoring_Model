import joblib
import pandas as pd
from feature_engineering import transform_with_preprocessor

MODEL_PATH = 'credit_scoring_model.pkl'
PRE_PATH = 'preprocessor.pkl'
CSV_PATH = 'loan_data.csv'
ACCT = 1200031067

model = joblib.load(MODEL_PATH)
pre = joblib.load(PRE_PATH)
df = pd.read_csv(CSV_PATH)
rec = df[df['AccountNumber'] == ACCT]
if rec.empty:
    print('Account not found')
else:
    X = rec.drop(['AccountNumber', 'Default', 'Payment History'], axis=1, errors='ignore')
    X_trans = transform_with_preprocessor(pre, X)
    pred = int(model.predict(X_trans)[0])
    label = 'Loan Approved(low risk)' if pred == 1 else 'Loan Denied(high risk)'
    print(f'Account: {ACCT}, Credit Score: {int(rec.iloc[0]["Credit Score"])}, Prediction: {pred}, Label: {label}')
