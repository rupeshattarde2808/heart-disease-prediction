
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
DATA_PATH = "dataset/heart_disease_uci.csv"
MODEL_DIR = "models"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# 2. Remove ID and convert target to binary
df = df.drop(columns=["id"])
df["num"] = (df["num"] > 0).astype(int)

# 3. Split features and target
X = df.drop(columns=["num"])
y = df["num"]

numeric_features = [
    "age", "trestbps", "chol", "thalch", "oldpeak", "ca"
]

categorical_features = [
    "sex", "dataset", "cp", "fbs",
    "restecg", "exang", "slope", "thal"
]

# 4. Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Create preprocessing pipelines
numeric_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_tree = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

def make_preprocessor(numeric_transformer):
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical, categorical_features)
    ])

# 6. Define the three models
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", make_preprocessor(numeric_lr)),
        ("classifier", LogisticRegression(
            max_iter=1000, random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", make_preprocessor(numeric_tree)),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("preprocessor", make_preprocessor(numeric_tree)),
        ("classifier", XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1
        ))
    ])
}

# 7. Create models folder if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# 8. Train, evaluate, and save each model
results = []

for name, model in models.items():
    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "name": name,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    })

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    save_path = os.path.join(
        MODEL_DIR, model_files[name]
    )
    joblib.dump(model, save_path)
    print("Saved model to:", save_path)

# 9. Compare model accuracy
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for result in sorted(
    results,
    key=lambda r: (r["recall"], r["f1"], r["accuracy"]),
    reverse=True
):
    print(f"\nModel: {result['name']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Recall:   {result['recall']:.4f}")
    print(f"F1-score: {result['f1']:.4f}")
    
print("\nTraining completed successfully.")


# Select the best-performing model
best_result = max(
    results,
    key=lambda r: (r["recall"], r["f1"], r["accuracy"])
)

best_model_name = best_result["name"]

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

print("\nBest model:", best_model_name)
print("Recall:", round(best_result["recall"], 4))
print("F1-score:", round(best_result["f1"], 4))
print("Accuracy:", round(best_result["accuracy"], 4))

with open("models/best_model.txt", "w") as file:
    file.write(model_files[best_model_name])