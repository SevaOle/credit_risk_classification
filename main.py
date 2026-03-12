import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/cleaned_data.csv")

df = df.drop(columns=[
    "Credit_Worthiness",
    "credit_type",
    "co-applicant_credit_type",
    "submission_of_application",
    "lump_sum_payment",
    "Neg_ammortization",
    "Interest_rate_spread",
    "Upfront_charges",
    "rate_of_interest"
])
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
print(confusion_matrix(labels_test, labels_pred))
print(classification_report(labels_test, labels_pred))



print("="*20)
print("\nRandom Forest:")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(features_train, labels_train)
rf_pred = rf_model.predict(features_test)


print(accuracy_score(labels_test, rf_pred))
print(confusion_matrix(labels_test, rf_pred))
print(classification_report(labels_test, rf_pred))

importance = pd.Series(rf_model.feature_importances_, index=features_train.columns)
print(importance.sort_values(ascending=False).head(10))