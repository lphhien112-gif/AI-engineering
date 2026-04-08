"""
📊 End-to-End sklearn Pipeline Demo
Chạy: pip install scikit-learn pandas numpy xgboost lightgbm
       python sklearn_pipeline.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("📊 sklearn Pipeline — Model Comparison Demo")
    print("=" * 60)

    # 1. Load data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    print(f"\n📋 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Target distribution: {dict(y.value_counts())}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Define models
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, probability=True)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]),
    }

    # Try XGBoost/LightGBM if available
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                eval_metric="logloss", random_state=42
            )),
        ])
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = Pipeline([
            ("clf", LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                verbose=-1, random_state=42
            )),
        ])
    except ImportError:
        pass

    # 4. Cross-Validation comparison
    print("\n🔄 5-Fold Cross-Validation Results:")
    print("-" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        results[name] = {
            "mean_f1": scores.mean(),
            "std_f1": scores.std(),
        }
        print(f"  {name:25s} → F1: {scores.mean():.4f} ± {scores.std():.4f}")

    # 5. Best model — train on full train set, evaluate on test
    best_name = max(results, key=lambda k: results[k]["mean_f1"])
    best_pipeline = models[best_name]

    print(f"\n🏆 Best Model: {best_name}")
    print("-" * 50)

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    # 6. Feature importance (if tree-based)
    if hasattr(best_pipeline.named_steps.get("clf", None), "feature_importances_"):
        importances = best_pipeline.named_steps["clf"].feature_importances_
        top_features = sorted(
            zip(data.feature_names, importances),
            key=lambda x: -x[1]
        )[:10]
        print("\n📊 Top 10 Feature Importances:")
        for name, imp in top_features:
            bar = "█" * int(imp * 100)
            print(f"  {name:30s} {imp:.4f} {bar}")

    print("\n" + "=" * 60)
    print("✅ Pipeline demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
