import pandas as pd

df = pd.read_csv('D:/My Projects/Predictive Maintainability RUL/artifacts/processed_tabular/test_tabular.csv')
if 'RUL' in df.columns:
    print('yes')
else:
    print('no')
#print(df['RUL'].head())
#print(df['RUL'].isnull().sum())