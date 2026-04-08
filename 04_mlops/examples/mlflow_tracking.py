"""
📊 MLflow Tracking Demo — Experiment tracking simulation
Chạy: python mlflow_tracking.py

Simulates MLflow tracking patterns without requiring MLflow server.
"""
import json
import os
import time
import hashlib
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class Run:
    run_id: str
    run_name: str
    experiment: str
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)
    tags: dict = field(default_factory=dict)
    start_time: str = ""
    end_time: str = ""
    status: str = "RUNNING"


class SimpleMLflowTracker:
    """Mimics MLflow tracking API for educational purposes."""
    
    def __init__(self, tracking_dir: str = "./mlruns"):
        self.tracking_dir = tracking_dir
        self.experiments: dict[str, list[Run]] = {}
        self.active_run: Run | None = None
    
    def set_experiment(self, name: str):
        if name not in self.experiments:
            self.experiments[name] = []
        self._current_experiment = name
    
    def start_run(self, run_name: str = None) -> Run:
        run_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        run_name = run_name or f"run-{run_id}"
        
        self.active_run = Run(
            run_id=run_id,
            run_name=run_name,
            experiment=self._current_experiment,
            start_time=datetime.now().isoformat(),
        )
        return self.active_run
    
    def log_param(self, key: str, value):
        self.active_run.params[key] = value
    
    def log_params(self, params: dict):
        self.active_run.params.update(params)
    
    def log_metric(self, key: str, value: float, step: int = None):
        if key not in self.active_run.metrics:
            self.active_run.metrics[key] = []
        self.active_run.metrics[key].append({"value": value, "step": step})
    
    def log_metrics(self, metrics: dict, step: int = None):
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def log_artifact(self, path: str):
        self.active_run.artifacts.append(path)
    
    def set_tag(self, key: str, value: str):
        self.active_run.tags[key] = value
    
    def end_run(self):
        self.active_run.end_time = datetime.now().isoformat()
        self.active_run.status = "FINISHED"
        self.experiments[self._current_experiment].append(self.active_run)
        run = self.active_run
        self.active_run = None
        return run
    
    def search_runs(self, experiment: str = None) -> list[Run]:
        exp = experiment or self._current_experiment
        return self.experiments.get(exp, [])
    
    def compare_runs(self) -> str:
        runs = self.search_runs()
        if not runs:
            return "No runs found."
        
        lines = []
        header = f"{'Run Name':<25} {'Status':<10}"
        metric_keys = set()
        param_keys = set()
        for run in runs:
            metric_keys.update(run.metrics.keys())
            param_keys.update(run.params.keys())
        
        metric_keys = sorted(metric_keys)
        param_keys = sorted(param_keys)
        
        header += " ".join(f"{k:<12}" for k in metric_keys)
        header += " | "
        header += " ".join(f"{k:<12}" for k in param_keys)
        lines.append(header)
        lines.append("-" * len(header))
        
        for run in runs:
            line = f"{run.run_name:<25} {run.status:<10}"
            for k in metric_keys:
                if k in run.metrics:
                    last_val = run.metrics[k][-1]["value"]
                    line += f"{last_val:<12.4f}"
                else:
                    line += f"{'N/A':<12}"
            line += " | "
            for k in param_keys:
                val = str(run.params.get(k, "N/A"))[:10]
                line += f"{val:<12}"
            lines.append(line)
        
        return "\n".join(lines)


def simulate_training(tracker: SimpleMLflowTracker, config: dict):
    """Simulate a training run with metrics logging."""
    import random
    random.seed(hash(str(config)) % 2**32)
    
    run = tracker.start_run(run_name=config["run_name"])
    tracker.log_params(config)
    tracker.set_tag("developer", "ai-engineer")
    tracker.set_tag("framework", "pytorch")
    
    print(f"\n  🏃 Run: {config['run_name']} (id: {run.run_id})")
    
    # Simulate training epochs
    base_acc = 0.7 + random.random() * 0.15
    lr = config.get("lr", 0.001)
    
    for epoch in range(config.get("epochs", 10)):
        # Simulated metrics
        train_loss = 1.0 / (epoch + 1) + random.random() * 0.1
        val_loss = 1.0 / (epoch + 1) + random.random() * 0.15
        accuracy = min(base_acc + epoch * 0.015 + random.random() * 0.01, 0.99)
        
        tracker.log_metrics({
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "accuracy": round(accuracy, 4),
        }, step=epoch)
        
        if epoch % 3 == 0 or epoch == config.get("epochs", 10) - 1:
            print(f"    Epoch {epoch+1:2d}: loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, acc={accuracy:.4f}")
    
    # Log artifacts
    tracker.log_artifact("confusion_matrix.png")
    tracker.log_artifact("model.pth")
    
    tracker.end_run()
    print(f"  ✅ Run completed: accuracy={accuracy:.4f}")


def main():
    print("=" * 60)
    print("📊 MLflow Experiment Tracking Demo")
    print("=" * 60)
    
    tracker = SimpleMLflowTracker()
    tracker.set_experiment("customer-churn-prediction")
    
    # Multiple experiment runs
    configs = [
        {
            "run_name": "rf-baseline",
            "model": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10,
            "lr": 0.01,
            "epochs": 10,
        },
        {
            "run_name": "xgboost-tuned",
            "model": "XGBoost",
            "n_estimators": 200,
            "max_depth": 6,
            "lr": 0.05,
            "epochs": 15,
        },
        {
            "run_name": "lgbm-optimized",
            "model": "LightGBM",
            "n_estimators": 300,
            "max_depth": 8,
            "lr": 0.03,
            "epochs": 12,
        },
        {
            "run_name": "nn-deep",
            "model": "NeuralNet",
            "hidden_size": 128,
            "dropout": 0.3,
            "lr": 0.001,
            "epochs": 20,
        },
    ]
    
    print("\n📋 Running experiments...")
    for config in configs:
        simulate_training(tracker, config)
    
    # Compare runs
    print(f"\n{'='*60}")
    print("📊 Experiment Comparison")
    print("=" * 60)
    print(tracker.compare_runs())
    
    # Find best run
    runs = tracker.search_runs()
    best_run = max(runs, key=lambda r: r.metrics.get("accuracy", [{"value": 0}])[-1]["value"])
    best_acc = best_run.metrics["accuracy"][-1]["value"]
    
    print(f"\n🏆 Best Run: {best_run.run_name}")
    print(f"   Accuracy: {best_acc:.4f}")
    print(f"   Model: {best_run.params.get('model')}")
    print(f"   Artifacts: {best_run.artifacts}")
    
    # Model Registry simulation
    print(f"\n📦 Model Registry:")
    print(f"   Registering {best_run.run_name} as 'churn-model' v1")
    print(f"   Stage: None → Staging → Production")
    
    # MLflow code template
    print(f"\n{'='*60}")
    print("💡 Real MLflow Code Template:")
    print("=" * 60)
    print("""
    import mlflow
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("my-experiment")
    mlflow.autolog()  # Auto-track everything!
    
    with mlflow.start_run(run_name="my-run"):
        model.fit(X_train, y_train)
        # Everything auto-logged!
    
    # View UI: mlflow ui --port 5000
    """)
    
    print(f"{'='*60}")
    print("✅ MLflow Tracking Demo completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
