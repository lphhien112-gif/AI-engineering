"""
📦 DVC Pipeline Demo — Data versioning concepts
Chạy: python dvc_pipeline.py

Simulates DVC pipeline stages: prepare → train → evaluate.
"""
import json
import hashlib
import os
import time
from datetime import datetime
import numpy as np


class DvcSimulator:
    """Simulate DVC data versioning and pipeline concepts."""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = project_dir
        self.stages = {}
        self.cache = {}
        self.metrics = {}
    
    def add(self, filepath: str, content: str = None):
        """Simulate 'dvc add' — create .dvc tracking file."""
        file_hash = hashlib.md5((content or filepath).encode()).hexdigest()[:12]
        dvc_file = {
            "md5": file_hash,
            "outs": [{"md5": file_hash, "path": filepath, "size": len(content or "")}],
        }
        print(f"  📦 dvc add {filepath}")
        print(f"     → Created {filepath}.dvc (hash: {file_hash})")
        print(f"     → Added {filepath} to .gitignore")
        return dvc_file
    
    def add_stage(self, name: str, cmd: str, deps: list, outs: list,
                  params: list = None, metrics: list = None):
        """Define a pipeline stage."""
        self.stages[name] = {
            "cmd": cmd, "deps": deps, "outs": outs,
            "params": params or [], "metrics": metrics or [],
        }
    
    def repro(self, stage: str = None):
        """Simulate 'dvc repro' — run/re-run pipeline stages."""
        stages_to_run = [stage] if stage else list(self.stages.keys())
        
        for stage_name in stages_to_run:
            stage_def = self.stages[stage_name]
            deps_hash = hashlib.md5(str(stage_def["deps"]).encode()).hexdigest()[:8]
            
            if deps_hash in self.cache:
                print(f"  ⏭️  Stage '{stage_name}': unchanged (cached)")
                continue
            
            print(f"  ▶️  Stage '{stage_name}': running...")
            print(f"     cmd: {stage_def['cmd']}")
            print(f"     deps: {stage_def['deps']}")
            print(f"     outs: {stage_def['outs']}")
            
            self.cache[deps_hash] = True
            time.sleep(0.2)
            print(f"  ✅ Stage '{stage_name}': completed")


def demo_dvc_basics():
    """Demonstrate DVC basics."""
    print("=== 1. DVC Basics ===\n")
    
    dvc = DvcSimulator()
    
    print("📋 Tracking a dataset:")
    dvc.add("data/training_images/", "10GB of labeled images")
    
    print(f"\n📋 Git workflow:")
    print(f"  $ git add data/training_images.dvc .gitignore")
    print(f"  $ git commit -m 'feat: add training dataset v1'")
    print(f"  $ dvc push  # Upload data to remote storage (S3/GCS)")
    
    print(f"\n📋 On another machine:")
    print(f"  $ git clone <repo>")
    print(f"  $ dvc pull  # Download data from remote storage")


def demo_dvc_pipeline():
    """Demonstrate DVC pipeline."""
    print("\n=== 2. DVC Pipeline (dvc.yaml) ===\n")
    
    dvc = DvcSimulator()
    
    # Define stages
    dvc.add_stage(
        "prepare",
        cmd="python src/prepare.py",
        deps=["src/prepare.py", "data/raw/"],
        outs=["data/processed/"],
        params=["prepare.split_ratio", "prepare.seed"],
    )
    
    dvc.add_stage(
        "train",
        cmd="python src/train.py",
        deps=["src/train.py", "data/processed/"],
        outs=["models/model.pth"],
        params=["train.model", "train.lr", "train.epochs"],
        metrics=["metrics/train_metrics.json"],
    )
    
    dvc.add_stage(
        "evaluate",
        cmd="python src/evaluate.py",
        deps=["src/evaluate.py", "models/model.pth"],
        outs=[],
        metrics=["metrics/eval_metrics.json"],
    )
    
    # Show dvc.yaml
    print("📋 dvc.yaml:")
    for name, stage in dvc.stages.items():
        print(f"  {name}:")
        print(f"    cmd: {stage['cmd']}")
        print(f"    deps: {stage['deps']}")
        print(f"    outs: {stage['outs']}")
        if stage['metrics']:
            print(f"    metrics: {stage['metrics']}")
    
    # Run pipeline
    print(f"\n📋 Running pipeline (dvc repro):")
    dvc.repro()
    
    # Second run — cached
    print(f"\n📋 Re-running (nothing changed):")
    dvc.repro()


def demo_params():
    """Demonstrate params.yaml."""
    print("\n=== 3. Params Configuration ===\n")
    
    params = {
        "prepare": {
            "split_ratio": 0.8,
            "seed": 42,
            "img_size": 768,
        },
        "train": {
            "model": "segformer-b5",
            "backbone": "mit_b5",
            "lr": 0.0001,
            "weight_decay": 0.01,
            "epochs": 50,
            "batch_size": 8,
            "loss": "focal_dice",
        },
        "evaluate": {
            "threshold": 0.5,
            "metrics": ["miou", "dice", "accuracy"],
        },
    }
    
    print("📋 params.yaml:")
    for section, values in params.items():
        print(f"  {section}:")
        for key, value in values.items():
            print(f"    {key}: {value}")


def demo_experiments():
    """Demonstrate DVC experiments."""
    print("\n=== 4. DVC Experiments ===\n")
    
    experiments = [
        {"name": "exp-baseline", "lr": 0.001, "model": "resnet18", "accuracy": 0.891, "f1": 0.873},
        {"name": "exp-lr-low", "lr": 0.0001, "model": "resnet18", "accuracy": 0.912, "f1": 0.901},
        {"name": "exp-resnet50", "lr": 0.0005, "model": "resnet50", "accuracy": 0.935, "f1": 0.928},
        {"name": "exp-segformer", "lr": 0.0001, "model": "segformer-b5", "accuracy": 0.958, "f1": 0.951},
    ]
    
    print("📋 dvc exp show:")
    print(f"  {'Experiment':<18} {'accuracy':<12} {'f1':<12} {'lr':<12} {'model'}")
    print(f"  {'-'*65}")
    
    best_exp = None
    best_acc = 0
    
    for exp in experiments:
        marker = ""
        if exp["accuracy"] > best_acc:
            best_acc = exp["accuracy"]
            best_exp = exp
            marker = " ⭐"
        print(f"  {exp['name']:<18} {exp['accuracy']:<12.3f} {exp['f1']:<12.3f} "
              f"{exp['lr']:<12} {exp['model']}{marker}")
    
    print(f"\n  🏆 Best: {best_exp['name']} (accuracy: {best_exp['accuracy']:.3f})")
    print(f"\n  Commands:")
    print(f"    $ dvc exp run --set-param train.lr=0.0001")
    print(f"    $ dvc exp apply {best_exp['name']}  # Apply best experiment")
    print(f"    $ git commit -m 'experiment: {best_exp['model']} lr={best_exp['lr']}'")


def demo_comparison():
    """Compare DVC with alternatives."""
    print("\n=== 5. Data Versioning Comparison ===\n")
    
    tools = [
        ("DVC", "Free/OSS", "S3/GCS/local", "Yes", "CLI", "Best overall"),
        ("Git LFS", "Free", "Git server", "No", "Git", "Simple, limited"),
        ("DagsHub", "Freemium", "DagsHub cloud", "DVC compat", "Web", "Easy DVC hosting"),
        ("LakeFS", "Free/OSS", "S3-compatible", "No", "Git-like", "Data lake versioning"),
        ("Delta Lake", "Free/OSS", "Object storage", "No", "Spark", "Big data versioning"),
    ]
    
    print(f"  {'Tool':<12} {'Price':<12} {'Storage':<16} {'Pipeline':<12} {'UI':<8} {'Notes'}")
    print(f"  {'-'*72}")
    for name, price, storage, pipeline, ui, notes in tools:
        print(f"  {name:<12} {price:<12} {storage:<16} {pipeline:<12} {ui:<8} {notes}")


def main():
    print("=" * 60)
    print("📦 DVC Data Versioning Demo")
    print("=" * 60)
    
    demo_dvc_basics()
    demo_dvc_pipeline()
    demo_params()
    demo_experiments()
    demo_comparison()
    
    print(f"\n{'='*60}")
    print("✅ DVC Pipeline Demo completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
