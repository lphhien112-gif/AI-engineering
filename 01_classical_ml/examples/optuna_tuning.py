"""
⚡ Optuna Hyperparameter Tuning Demo
Chạy: pip install optuna scikit-learn xgboost
       python optuna_tuning.py
"""
import optuna
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    print("=" * 60)
    print("⚡ Optuna Hyperparameter Tuning Demo")
    print("=" * 60)

    # Load data
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Baseline (no tuning) ---
    baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_scores = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="f1")
    print(f"\n📊 Baseline RF: F1 = {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}")

    # --- Optuna objective ---
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }

        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        score = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1").mean()
        return score

    # --- Run optimization ---
    print("\n🔍 Running Optuna optimization (50 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\n🏆 Best F1: {study.best_value:.4f}")
    print(f"   Best params:")
    for key, value in study.best_params.items():
        print(f"     {key}: {value}")

    # --- Improvement ---
    improvement = (study.best_value - baseline_scores.mean()) / baseline_scores.mean() * 100
    print(f"\n📈 Improvement over baseline: {improvement:+.2f}%")

    # --- Final evaluation on test set ---
    best_clf = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
    best_clf.fit(X_train, y_train)
    test_score = best_clf.score(X_test, y_test)
    print(f"🎯 Test Accuracy: {test_score:.4f}")

    # --- Parameter importance ---
    print("\n📊 Parameter Importance:")
    importances = optuna.importance.get_param_importances(study)
    for param, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"  {param:25s} {imp:.4f} {bar}")

    # --- Optimization history ---
    print("\n📈 Optimization History (best value over trials):")
    best_so_far = float("-inf")
    milestones = [0, 9, 19, 29, 39, 49]
    for i, trial in enumerate(study.trials):
        if trial.value and trial.value > best_so_far:
            best_so_far = trial.value
        if i in milestones:
            print(f"  Trial {i+1:3d}: Best F1 = {best_so_far:.4f}")

    print("\n" + "=" * 60)
    print("✅ Optuna tuning completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
