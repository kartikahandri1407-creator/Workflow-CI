import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="namadataset_preprocessing/heart_preprocessing.csv")
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

df = pd.read_csv(args.data_path)
X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, "model")

    print("Akurasi:", accuracy_score(y_test, y_pred))