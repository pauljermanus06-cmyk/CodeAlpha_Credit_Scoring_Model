import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def add_basic_features(df):
    df = df.copy()
    if 'Loan Amount' in df.columns and 'Income' in df.columns:
        df['loan_to_income'] = df['Loan Amount'] / (df['Income'] + 1)
    if 'Credit Score' in df.columns:
        bins = [0, 579, 669, 739, 799, 850]
        labels = ['very_poor', 'poor', 'fair', 'good', 'excellent']
        df['credit_score_bucket'] = pd.cut(df['Credit Score'], bins=bins, labels=labels)
    if 'Age' in df.columns:
        age_bins = [17, 25, 35, 50, 120]
        age_labels = ['18-25', '26-35', '36-50', '51+']
        df['age_bucket'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    return df


def build_preprocessor(X):
    X = X.copy()
    # detect numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # exclude boolean/int target-like columns if present externally
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # numeric pipeline: median imputation -> log1p -> standard scaling
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, validate=True)),
        ('scaler', StandardScaler())
    ])

    # categorical pipeline: most frequent impute -> one-hot
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers, remainder='drop')
    return preprocessor


def fit_transformer_on_train(X_train):
    X_train_fe = add_basic_features(X_train)
    preprocessor = build_preprocessor(X_train_fe)
    X_train_transformed = preprocessor.fit_transform(X_train_fe)
    return preprocessor, X_train_transformed


def transform_with_preprocessor(preprocessor, X):
    X_fe = add_basic_features(X)
    return preprocessor.transform(X_fe)


def compute_recent_spend(users, months=3, include_statuses=('paid',)):
    """Compute total amount spent per user in the last `months` months.

    users: list of dicts, each with keys 'user_id' and 'payment_history' (list of dicts with 'date','amount','status')
    returns: DataFrame indexed by user_id with column `recent_spend_{months}m`
    """
    from datetime import datetime, timedelta

    cutoff = datetime.today() - timedelta(days=30 * months)
    rows = []
    for u in users:
        total = 0.0
        for p in u.get('payment_history', []):
            date_str = p.get('date')
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                try:
                    d = pd.to_datetime(date_str).to_pydatetime()
                except Exception:
                    continue
            if d >= cutoff and (p.get('status') in include_statuses if include_statuses is not None else True):
                try:
                    total += float(p.get('amount', 0))
                except Exception:
                    continue
        rows.append({'user_id': u.get('user_id'), f'recent_spend_{months}m': round(total, 2)})

    df = pd.DataFrame(rows).set_index('user_id')
    return df
