import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


def train_model():
    print("Loading cleaned dataset...")
    df = pd.read_csv("dubizzle_laptops_cleaned.csv")

    y = df["Market_Tier"]
    X_num = df[
        [
            "RAM_GB",
            "Storage_GB",
            "GPU_Tier",
            "Damaged_Display",
            "Body_Damage",
            "Missing_Charger",
            "Perfect_Condition",
            "Arabic_Keyboard",
        ]
    ]
    X_cat = pd.get_dummies(
        df[["Market_Segment", "CPU_Tier", "Screen_Size"]], drop_first=True
    )
    X_final = pd.concat([X_num, X_cat], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.15, random_state=42
    )

    print("Training V2 XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"ACCURACY: {accuracy_score(y_test, preds) * 100:.2f}%\n")
    print(
        classification_report(
            y_test, preds, target_names=["Budget", "Mid-Range", "Premium"]
        )
    )

    joblib.dump(model, "valuation_engine.joblib")
    joblib.dump(X_final.columns.tolist(), "model_columns.joblib")
    print("Exported valuation_engine.joblib and model_columns.joblib")


if __name__ == "__main__":
    train_model()
