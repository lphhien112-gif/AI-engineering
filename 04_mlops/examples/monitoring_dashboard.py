"""
📈 Model Monitoring Dashboard Demo
Chạy: python monitoring_dashboard.py

Simulates production model monitoring: drift detection, alerts, metrics.
"""
import numpy as np
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class PredictionLog:
    timestamp: str
    input_features: dict
    prediction: float
    confidence: float
    latency_ms: float
    ground_truth: float = None


class DriftDetector:
    """Detect data drift using statistical methods."""
    
    def __init__(self, reference_data: np.ndarray):
        self.reference = reference_data
        self.ref_mean = np.mean(reference_data)
        self.ref_std = np.std(reference_data)
    
    def ks_test(self, production_data: np.ndarray) -> dict:
        """Kolmogorov-Smirnov test."""
        n1 = len(self.reference)
        n2 = len(production_data)
        
        all_data = np.concatenate([self.reference, production_data])
        all_data.sort()
        
        cdf1 = np.searchsorted(np.sort(self.reference), all_data, side='right') / n1
        cdf2 = np.searchsorted(np.sort(production_data), all_data, side='right') / n2
        
        ks_stat = np.max(np.abs(cdf1 - cdf2))
        # Approximate p-value
        n_eff = (n1 * n2) / (n1 + n2)
        p_value = np.exp(-2 * n_eff * ks_stat ** 2)
        
        return {
            "test": "Kolmogorov-Smirnov",
            "statistic": round(ks_stat, 4),
            "p_value": round(p_value, 4),
            "drift": p_value < 0.05,
            "severity": "none" if p_value > 0.1 else "moderate" if p_value > 0.01 else "severe",
        }
    
    def psi(self, production_data: np.ndarray, bins: int = 10) -> dict:
        """Population Stability Index."""
        breakpoints = np.percentile(self.reference, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        ref_counts = np.histogram(self.reference, bins=breakpoints)[0] + 1
        prod_counts = np.histogram(production_data, bins=breakpoints)[0] + 1
        
        ref_pct = ref_counts / ref_counts.sum()
        prod_pct = prod_counts / prod_counts.sum()
        
        psi_value = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
        
        return {
            "test": "PSI",
            "psi": round(psi_value, 4),
            "drift": psi_value > 0.2,
            "interpretation": (
                "No significant change" if psi_value < 0.1
                else "Moderate shift" if psi_value < 0.2
                else "Significant shift — investigate!"
            ),
        }
    
    def mean_shift(self, production_data: np.ndarray) -> dict:
        """Simple mean/std comparison."""
        prod_mean = np.mean(production_data)
        prod_std = np.std(production_data)
        
        mean_shift = abs(prod_mean - self.ref_mean) / self.ref_std
        std_ratio = prod_std / self.ref_std
        
        return {
            "test": "Mean/Std Comparison",
            "ref_mean": round(self.ref_mean, 4),
            "prod_mean": round(prod_mean, 4),
            "mean_shift_std": round(mean_shift, 2),
            "std_ratio": round(std_ratio, 2),
            "drift": mean_shift > 2.0 or std_ratio > 1.5 or std_ratio < 0.67,
        }


class ModelMonitor:
    """Production model monitoring system."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logs: list[PredictionLog] = []
        self.alerts: list[dict] = []
    
    def log_prediction(self, features: dict, prediction: float,
                       confidence: float, latency_ms: float):
        self.logs.append(PredictionLog(
            timestamp=datetime.now().isoformat(),
            input_features=features,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
        ))
    
    def get_dashboard(self) -> dict:
        """Generate monitoring dashboard metrics."""
        if not self.logs:
            return {"status": "no_data"}
        
        latencies = [log.latency_ms for log in self.logs]
        confidences = [log.confidence for log in self.logs]
        predictions = [log.prediction for log in self.logs]
        
        return {
            "model": self.model_name,
            "total_predictions": len(self.logs),
            "time_range": f"{self.logs[0].timestamp[:19]} → {self.logs[-1].timestamp[:19]}",
            "latency": {
                "p50": round(np.percentile(latencies, 50), 1),
                "p95": round(np.percentile(latencies, 95), 1),
                "p99": round(np.percentile(latencies, 99), 1),
                "mean": round(np.mean(latencies), 1),
            },
            "confidence": {
                "mean": round(np.mean(confidences), 3),
                "below_50pct": sum(1 for c in confidences if c < 0.5),
                "below_70pct": sum(1 for c in confidences if c < 0.7),
            },
            "predictions": {
                "mean": round(np.mean(predictions), 4),
                "std": round(np.std(predictions), 4),
                "positive_rate": round(np.mean(np.array(predictions) > 0.5), 3),
            },
        }
    
    def check_alerts(self) -> list[dict]:
        """Check for alert conditions."""
        alerts = []
        dashboard = self.get_dashboard()
        
        if dashboard.get("status") == "no_data":
            return [{"level": "warning", "message": "No predictions logged"}]
        
        latency = dashboard["latency"]
        if latency["p95"] > 200:
            alerts.append({
                "level": "warning",
                "metric": "latency_p95",
                "value": latency["p95"],
                "threshold": 200,
                "message": f"High latency: P95={latency['p95']}ms (threshold: 200ms)",
            })
        
        conf = dashboard["confidence"]
        low_conf_rate = conf["below_70pct"] / dashboard["total_predictions"]
        if low_conf_rate > 0.1:
            alerts.append({
                "level": "critical",
                "metric": "low_confidence_rate",
                "value": round(low_conf_rate, 3),
                "threshold": 0.1,
                "message": f"High rate of low-confidence predictions: {low_conf_rate:.1%}",
            })
        
        self.alerts.extend(alerts)
        return alerts


def main():
    print("=" * 60)
    print("📈 Model Monitoring Dashboard Demo")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 1. Drift Detection
    print("\n=== 1. Drift Detection ===\n")
    
    # Reference (training) distribution
    reference = np.random.normal(loc=5.0, scale=2.0, size=5000)
    
    # Scenario A: No drift
    production_a = np.random.normal(loc=5.1, scale=2.1, size=1000)
    # Scenario B: Moderate drift
    production_b = np.random.normal(loc=6.0, scale=2.5, size=1000)
    # Scenario C: Severe drift
    production_c = np.random.normal(loc=8.0, scale=3.0, size=1000)
    
    detector = DriftDetector(reference)
    
    scenarios = [
        ("No Drift", production_a),
        ("Moderate Drift", production_b),
        ("Severe Drift", production_c),
    ]
    
    for name, prod_data in scenarios:
        print(f"  📊 Scenario: {name}")
        
        ks = detector.ks_test(prod_data)
        psi = detector.psi(prod_data)
        mean = detector.mean_shift(prod_data)
        
        print(f"     KS: stat={ks['statistic']}, p={ks['p_value']}, "
              f"drift={'🔴' if ks['drift'] else '🟢'} ({ks['severity']})")
        print(f"     PSI: {psi['psi']}, drift={'🔴' if psi['drift'] else '🟢'} "
              f"({psi['interpretation']})")
        print(f"     Mean shift: {mean['mean_shift_std']}σ, "
              f"drift={'🔴' if mean['drift'] else '🟢'}")
        print()
    
    # 2. Model Monitoring
    print("=== 2. Production Monitoring ===\n")
    
    monitor = ModelMonitor("churn-prediction-v2")
    
    # Simulate 500 predictions
    print("  Simulating 500 production predictions...")
    for i in range(500):
        features = {
            "age": np.random.randint(18, 70),
            "tenure": np.random.randint(1, 60),
            "monthly_charges": round(np.random.uniform(20, 120), 2),
        }
        
        prediction = round(np.random.beta(2, 5), 4)
        confidence = round(np.random.beta(5, 2), 3)
        latency = round(np.random.lognormal(3, 0.5), 1)  # ~20-100ms
        
        # Inject some anomalies
        if i > 400:
            latency *= 3  # Latency spike
            confidence *= 0.5  # Confidence drop
        
        monitor.log_prediction(features, prediction, confidence, latency)
    
    # Display dashboard
    dashboard = monitor.get_dashboard()
    print(f"\n  📊 Dashboard: {dashboard['model']}")
    print(f"     Total predictions: {dashboard['total_predictions']}")
    print(f"     Time range: {dashboard['time_range']}")
    print(f"\n     ⏱️  Latency:")
    for k, v in dashboard['latency'].items():
        print(f"        {k}: {v}ms")
    print(f"\n     🎯 Confidence:")
    for k, v in dashboard['confidence'].items():
        print(f"        {k}: {v}")
    print(f"\n     📈 Predictions:")
    for k, v in dashboard['predictions'].items():
        print(f"        {k}: {v}")
    
    # Check alerts
    print(f"\n=== 3. Alerts ===\n")
    alerts = monitor.check_alerts()
    if alerts:
        for alert in alerts:
            icon = "🚨" if alert["level"] == "critical" else "⚠️"
            print(f"  {icon} [{alert['level'].upper()}] {alert['message']}")
    else:
        print("  ✅ No alerts — all metrics within normal range")
    
    # Retraining triggers
    print(f"\n=== 4. Retraining Triggers ===\n")
    triggers = [
        ("Scheduled", "Monthly", "✅ Due in 12 days"),
        ("Performance", "Accuracy < 90%", "🟢 Current: 93.2%"),
        ("Data Drift", "PSI > 0.2", f"{'🔴' if psi['drift'] else '🟢'} PSI: {psi['psi']}"),
        ("Data Volume", "10K+ new labels", "🟢 Current: 3.2K"),
        ("Latency", "P95 > 200ms", f"{'🔴' if dashboard['latency']['p95'] > 200 else '🟢'} P95: {dashboard['latency']['p95']}ms"),
    ]
    
    for trigger, condition, status in triggers:
        print(f"  {status} {trigger}: {condition}")
    
    print(f"\n{'='*60}")
    print("✅ Model Monitoring Demo completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
