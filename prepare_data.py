import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("data/Loan_Default.csv")

df = df.drop(columns=["ID"])
df = df.drop_duplicates()

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object", "string"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

df.to_csv("data/cleaned_data.csv", index=False)
df = pd.read_csv("data/cleaned_data.csv")