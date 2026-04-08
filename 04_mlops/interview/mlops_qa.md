# 🎯 MLOps — Câu Hỏi Phỏng Vấn (30+)

---

## Experiment Tracking (6 câu)

### Q1: MLflow components?
**A**: (1) Tracking: log params/metrics/artifacts, (2) Projects: packaging code, (3) Models: model format/serving, (4) Model Registry: staging/production lifecycle.

### Q2: MLflow vs W&B?
**A**: MLflow: open source, self-hosted, enterprise. W&B: cloud-managed, better UI, team collaboration. Both good — W&B cho research teams, MLflow cho enterprise self-hosted.

### Q3: Autolog hoạt động thế nào?
**A**: `mlflow.autolog()` monkey-patches training frameworks (sklearn, PyTorch, TF). Auto-records params, metrics, model artifacts. Zero-effort experiment tracking.

### Q4: Model Registry workflow?
**A**: Train → Log model → Register → Staging → (validate) → Production → (retire) → Archived. Each version tracked with metadata and lineage.

### Q5: Experiment reproducibility cần gì?
**A**: (1) Code version (git hash), (2) Data version (DVC hash), (3) Parameters (logged), (4) Environment (requirements.txt/conda.yaml), (5) Random seeds.

### Q6: Metric logging best practices?
**A**: Log per-epoch (not just final), include validation + training, log learning rate, log system metrics (GPU util, memory). Use consistent naming conventions.

---

## Data Versioning (5 câu)

### Q7: DVC là gì?
**A**: Data Version Control — "Git for data". Track large files with .dvc pointer files in git. Actual data stored in remote storage (S3, GCS, local). `dvc push/pull` like `git push/pull`.

### Q8: DVC Pipeline?
**A**: DAG of stages defined in dvc.yaml. Each stage: cmd + deps + outs + params. `dvc repro` only re-runs stages with changed dependencies. Ensures reproducibility.

### Q9: DVC vs Git LFS?
**A**: Git LFS: stores in git server, limited storage, no pipelines. DVC: any storage backend, pipeline support, experiment tracking, params management. DVC >> Git LFS cho ML.

### Q10: params.yaml dùng để làm gì?
**A**: Central config file. DVC stages reference params. Change param → DVC knows which stage to re-run. Experiment tracking logs params automatically.

### Q11: dvc repro tại sao hữu ích?
**A**: Chỉ re-run stages có dependencies thay đổi. Ví dụ: đổi model config → chỉ re-run train + evaluate, không re-run prepare. Tiết kiệm thời gian.

---

## Containerization (5 câu)

### Q12: Multi-stage Docker build?
**A**: Stage 1: install deps (heavy, build tools). Stage 2: copy only needed artifacts. Result: smaller image (2GB → 500MB). Critical for deployment.

### Q13: GPU in Docker?
**A**: NVIDIA Container Toolkit + `--gpus all` flag. Base image: `nvidia/cuda:12.1-runtime`. K8s: `nvidia.com/gpu` resource request.

### Q14: Docker Compose cho ML stack?
**A**: Multi-service: model API + MLflow + Redis + Prometheus. Docker Compose orchestrates locally. Production: migrate to K8s.

### Q15: K8s cho ML khi nào?
**A**: Multiple models, auto-scaling needed, GPU scheduling, rolling updates, high availability. Overkill for single-model small teams.

### Q16: Health checks trong ML containers?
**A**: Readiness: model loaded and ready for inference. Liveness: process running, not OOM. K8s probes: HTTP GET /health. Required for production.

---

## CI/CD (5 câu)

### Q17: CI/CD cho ML khác gì CI/CD thường?
**A**: + Data versioning (DVC), + model validation (quality gates), + metric comparison, + A/B testing, + canary deployment, + data testing.

### Q18: Quality gates?
**A**: Automated checks before deployment: accuracy > threshold, latency < budget, no regression vs previous version. Block deploy if any check fails.

### Q19: ML testing types?
**A**: (1) Unit: function correctness, (2) Data: schema, no NaN, distribution, (3) Model: shape, range, determinism, (4) Integration: full pipeline, (5) Performance: latency, throughput.

### Q20: Canary deployment?
**A**: Deploy new model to small % traffic (e.g., 5%). Monitor metrics. If good → gradually increase to 100%. If bad → rollback immediately. Safer than big-bang deployment.

### Q21: A/B testing models?
**A**: (1) Define metrics (business + model), (2) Split traffic, (3) Run sufficient time, (4) Statistical test, (5) Roll out winner. Need sufficient sample size for significance.

---

## Monitoring (5 câu)

### Q22: Model decay?
**A**: Performance degrades over time because production data diverges from training data. Causes: data drift, concept drift, schema changes. Solution: continuous monitoring + retraining.

### Q23: Data drift vs concept drift?
**A**: Data drift: P(X) changes (input distribution shifts). Concept drift: P(Y|X) changes (relationship changes). Both cause performance drop.

### Q24: PSI (Population Stability Index)?
**A**: Measures distribution shift between reference and production data. PSI < 0.1: stable. 0.1-0.2: moderate. > 0.2: significant drift. Action needed.

### Q25: Monitoring metrics?
**A**: (1) Prediction distribution, (2) Latency P50/P95/P99, (3) Error rate, (4) Feature drift (per feature), (5) Confidence distribution, (6) Business metrics (revenue, conversion).

### Q26: Retraining triggers?
**A**: (1) Scheduled (monthly), (2) Performance drop > threshold, (3) PSI > 0.2, (4) New labeled data > threshold, (5) Manual (feature update).

---

## Cloud (4 câu)

### Q27: Cloud platform nào cho ML?
**A**: GCP: best AI ecosystem (Vertex AI, TPU). AWS: biggest market (SageMaker). Azure: Microsoft integration. All viable — depends on existing cloud.

### Q28: Spot instances cho training?
**A**: 60-90% cheaper nhưng can be interrupted. Solution: checkpoint frequently → resume on new spot instance. Training only — never for serving.

### Q29: Serverless ML serving?
**A**: AWS Lambda, Cloud Run, SageMaker Serverless. Pay per request. Pros: zero cost at idle, auto-scale. Cons: cold starts, no GPU (usually), memory limits.

### Q30: Cost optimization?
**A**: (1) Spot training, (2) Auto-scale serving (scale-to-zero), (3) Model compression (INT8), (4) Right-size instances, (5) Scheduled scaling, (6) Reserved instances for baseline.
