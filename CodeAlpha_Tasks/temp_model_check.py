import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'C:\Users\PAUL JERMANUS\Python\CodeAlpha_Tasks\loan_data.csv')
print('shape', df.shape)
print(df.head())
print('\nDefault distribution:')
print(df['Default'].value_counts(dropna=False))
x = df.drop(['Default'], axis=1)
y = df['Default']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('\nTest prediction distribution:')
print(pd.Series(y_pred).value_counts())
print('\nAccuracy:', accuracy_score(y_test, y_pred))
print('\nFirst test rows:')
print(pd.concat([x_test.reset_index(drop=True).head(), y_test.reset_index(drop=True).head(), pd.Series(y_pred, name="pred")[:5]], axis=1))
