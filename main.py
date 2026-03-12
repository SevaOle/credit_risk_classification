import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/cleaned_data.csv")

features = df.drop("Status", axis=1)
labels = df["Status"]

features_train, features_test, labels_train, labels_test = train_test_split(
    features,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

print(features_train.shape)
print(features_test.shape)

scaler = StandardScaler()

features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

log_model = LogisticRegression(max_iter=2000)
log_model.fit(features_train_scaled, labels_train)

labels_pred = log_model.predict(features_test_scaled)

print(accuracy_score(labels_test, labels_pred))