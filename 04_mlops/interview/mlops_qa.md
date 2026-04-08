# 🎯 MLOps — Câu Hỏi Phỏng Vấn (40+)

> Mỗi câu quan trọng có: giải thích → ví dụ thực tế → code/table → follow-up.

---

## Experiment Tracking (6 câu)

### Q1: MLflow components?
**A**: 4 core components:
- **Tracking**: log params/metrics/artifacts. Central experiment store
- **Projects**: packaging code (MLproject file, conda env) → reproducible runs
- **Models**: unified model format + serving (REST, batch, edge)
- **Model Registry**: version control cho models — staging → production → archived

```python
import mlflow

with mlflow.start_run(run_name="transformer-v3"):
    mlflow.log_params({"lr": 0.001, "epochs": 50, "model": "bert-base"})
    mlflow.log_metrics({"val_acc": 0.92, "val_loss": 0.31})
    mlflow.log_artifact("model.onnx")
    mlflow.pytorch.log_model(model, "model")  # Auto-log model
```
- **Follow-up**: "Autolog hoạt động thế nào?" → `mlflow.autolog()` monkey-patches sklearn/PyTorch/TF — tự log params, metrics, model artifacts. Zero-effort tracking.

### Q2: MLflow vs W&B — chọn cái nào?
**A**: 

| Tiêu chí | MLflow | W&B (Weights & Biases) |
|----------|--------|------------------------|
| **Hosting** | Self-hosted (free) hoặc Databricks | Cloud-managed (free tier 100GB) |
| **UI/UX** | Functional, basic | Polish, interactive charts, Tables |
| **Team collab** | Limited (sharing via server) | Built-in: reports, comments, team spaces |
| **Custom viz** | Matplotlib artifacts | Interactive dashboards, custom panels |
| **LLM tracking** | Basic (log prompts as text) | Native: prompt traces, token usage, eval tables |
| **Pricing** | Free (self-host) | Free personal, $50/user/mo team |
| **Integration** | Sklearn, PyTorch, TF, Spark | + HuggingFace, LangChain, OpenAI native |
| **Offline mode** | Local file store | W&B offline mode + sync later |

**Decision guide**: 
- Enterprise + data sovereignty → MLflow (self-host, control)
- Research team + rapid iteration → W&B (better UX, collab)
- Startup + cost-sensitive → MLflow local → migrate later
- **Follow-up**: "Có thể dùng cả 2?" → Có. W&B cho experiment viz + MLflow cho model registry/serving. Khá phổ biến.

### Q3: Experiment reproducibility cần gì?
**A**: 5 pillars — thiếu 1 cái là không reproduce được:
1. **Code version**: git commit hash (exact code state)
2. **Data version**: DVC hash hoặc dataset fingerprint
3. **Parameters**: logged (lr, batch_size, epochs, model config)
4. **Environment**: `requirements.txt` + Python version + CUDA version
5. **Random seeds**: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`

```python
# Production reproducibility setup
import torch, numpy as np, random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Slower but reproducible
    torch.backends.cudnn.benchmark = False
```
- **⚠️ Gotcha**: `torch.backends.cudnn.deterministic` makes training ~10% slower. Only use for final reported results.

### Q4: Model Registry workflow?
**A**: Train → Log model → Register → **Staging** → (validate on test set) → **Production** → (monitor) → **Archived**.
- Each version tracked with: metadata, lineage (which run), tags, description
- **Transition rules**: CI/CD checks (accuracy > threshold, latency < budget) before Staging → Production
- **Follow-up**: "Multiple models in production?" → A/B testing, shadow mode (new model receives traffic nhưng results không serve to user, chỉ log).

### Q5: Metric logging best practices?
**A**: 
- Log **per-epoch** (not just final) — see training dynamics
- Log **cả train + validation** — detect overfitting
- Log **learning rate** — verify scheduler working
- Log **system metrics**: GPU utilization, memory, throughput (samples/sec)
- **Naming convention**: `{split}_{metric}` → `train_loss`, `val_accuracy`, `test_f1`
- **⚠️ Don't**: log quá nhiều (mỗi batch) → slow UI. Log mỗi epoch hoặc mỗi N steps.

### Q6: Experiment comparison — cách so sánh 2 runs?
**A**: 
- **Quantitative**: metrics table (accuracy, F1, latency, model size)
- **Qualitative**: error analysis (which samples improved/degraded)
- **Statistical**: confidence intervals, paired t-test nếu multiple seeds
- **Cost**: training time × GPU cost, inference cost per 1K requests
- **Follow-up**: "Khi nào kết quả có statistical significance?" → Run 3-5 seeds, report mean ± std. T-test p < 0.05.

---

## Data Versioning (5 câu)

### Q7: DVC là gì? Tại sao cần?
**A**: Data Version Control — "Git for data". Giải quyết: Git không track được file lớn (models, datasets).
- `.dvc` pointer files in git → actual data in remote storage (S3, GCS, local)
- `dvc push/pull` sync data giống `git push/pull`
- **Khi nào cần**: dataset > 100MB, cần reproduce experiments, team share data

```bash
# Workflow
dvc init
dvc add data/training_set.parquet   # Creates .dvc pointer file
git add data/training_set.parquet.dvc .gitignore
git commit -m "data: add training set v1"
dvc push                            # Upload to remote (S3/GCS)
```

### Q8: DVC Pipeline?
**A**: DAG of stages defined in `dvc.yaml`. Each stage: `cmd` + `deps` + `outs` + `params`.
```yaml
stages:
  prepare:
    cmd: python prepare.py
    deps: [data/raw.csv, src/prepare.py]
    outs: [data/processed.parquet]
  train:
    cmd: python train.py
    deps: [data/processed.parquet, src/train.py]
    params: [lr, epochs, model_type]     # From params.yaml
    outs: [models/best.pth]
    metrics: [metrics.json:accuracy]     # Track metrics
```
`dvc repro` only re-runs stages with changed dependencies → tiết kiệm thời gian.

### Q9: DVC vs Git LFS?
**A**: 

| | DVC | Git LFS |
|-|-----|---------|
| **Storage** | Any backend (S3, GCS, Azure, SSH) | Git server only |
| **Pipelines** | ✅ Built-in DAG | ❌ |
| **Params** | ✅ params.yaml tracking | ❌ |
| **Experiment tracking** | ✅ `dvc exp` | ❌ |
| **Cost** | Free (you pay storage) | Limited free (GitHub 1GB) |

**Verdict**: Git LFS cho binary files đơn giản. DVC cho ML workflows.

### Q10: params.yaml dùng để làm gì?
**A**: Central config file. DVC stages reference params → change param → DVC knows which stage to re-run. MLflow/W&B auto-log params.

### Q11: dvc repro tại sao hữu ích?
**A**: Chỉ re-run stages có dependencies thay đổi. VD: đổi model config → chỉ re-run train + evaluate, không re-run prepare (data unchanged). Giống `make` cho ML.

---

## Containerization (5 câu)

### Q12: Multi-stage Docker build cho ML?
**A**: 
```dockerfile
# Stage 1: Build (heavy, compile deps)
FROM python:3.11 AS builder
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime (slim, production)
FROM python:3.11-slim
COPY --from=builder /install /usr/local
COPY src/ /app/src/
COPY models/ /app/models/
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0"]
```
Result: 2GB → 500MB. No build tools, pip, gcc in production → smaller attack surface.

### Q13: GPU in Docker?
**A**: NVIDIA Container Toolkit + `--gpus all` flag. Base image: `nvidia/cuda:12.1-runtime-ubuntu22.04`.
- K8s: `nvidia.com/gpu: 1` in resource requests
- **⚠️ Gotcha**: CUDA version in container MUST match host driver compatibility. Use `nvidia-smi` to check.

### Q14: Docker Compose cho ML stack?
**A**: Multi-service local: model API + MLflow + Redis (cache) + Prometheus (monitoring).
- Production: migrate to K8s hoặc Cloud Run.

### Q15: Kubernetes cho ML khi nào?
**A**: Multiple models + auto-scaling + GPU scheduling + rolling updates + high availability. **Overkill** cho single-model small teams. Consider Cloud Run / Modal / Replicate trước.

### Q16: Health checks trong ML containers?
**A**: 
- **Readiness**: model loaded AND warm (first inference done) → ready for traffic
- **Liveness**: process running, not OOM, not deadlocked
- Pattern: `GET /health` returns model version + status. K8s probes check periodically.

---

## CI/CD for ML (5 câu)

### Q17: CI/CD cho ML khác gì CI/CD thường?
**A**: Traditional CI/CD test code. ML CI/CD test **code + data + model quality**.

```yaml
# .github/workflows/ml_pipeline.yml
name: ML Pipeline
on: [push]
jobs:
  test:
    steps:
      - run: pytest tests/ -v                    # Code tests
      - run: python scripts/validate_data.py     # Data schema + distribution
  train:
    needs: test
    steps:
      - run: python train.py --config prod.yaml
      - run: python evaluate.py                  # Quality gates
  deploy:
    needs: train
    if: ${{ steps.evaluate.outputs.accuracy > 0.90 }}  # Gate!
    steps:
      - run: ./deploy.sh --canary 5%             # Start with 5% traffic
```

Key additions vs traditional: data validation, model quality gates, metric comparison vs baseline, canary deployment, A/B testing.

### Q18: Quality gates là gì?
**A**: Automated checks TRƯỚC deployment:
- accuracy > threshold (e.g., 0.90)
- latency P95 < budget (e.g., 100ms)  
- No regression vs previous production version
- Data quality passed (no drift detected)
- **Block deploy** if any check fails. Giống unit tests nhưng cho model quality.

### Q19: ML testing types?
**A**: 5 layers:
1. **Unit**: function correctness (preprocessing, postprocessing)
2. **Data**: schema validation, no NaN, distribution checks, freshness
3. **Model**: output shape, value range, determinism with same seed
4. **Integration**: full pipeline end-to-end (data → preprocess → predict → postprocess)
5. **Performance**: latency P50/P95/P99, throughput, memory usage under load

### Q20: Canary deployment?
**A**: Deploy new model to 5% traffic → monitor metrics (accuracy, latency, errors) → gradually increase (5% → 25% → 50% → 100%). If metrics drop → immediate rollback. Safer than big-bang.
- **Follow-up**: "Shadow mode khác gì canary?" → Shadow: new model nhận traffic nhưng results **không** serve cho user, chỉ log để compare. Zero risk.

### Q21: A/B testing models?
**A**: (1) Define business + model metrics, (2) Random traffic split, (3) Run sufficient time (statistical power), (4) Test significance, (5) Roll out winner.
- **⚠️ Common mistake**: chạy A/B test quá ngắn → kết quả không significant. Cần tính sample size trước.

---

## Monitoring (5 câu)

### Q22: Model decay — tại sao model production xấu dần?
**A**: Performance degrades vì production data **diverge** khỏi training data theo thời gian.

**Ví dụ thực tế**: E-commerce recommendation model
- Tháng 1 (launch): CTR = 12% ✅ (trained on 2024 data)
- Tháng 6: CTR = 8% ⚠️ (new product categories, trends changed)
- Tháng 12: CTR = 5% 🔴 (user behavior completely different)

**Causes**: data drift, concept drift, feature pipeline bugs, upstream schema changes.
**Solution**: continuous monitoring + retraining triggers + versioned data pipelines.

### Q23: Data drift vs concept drift?
**A**: 

| | Data Drift | Concept Drift |
|-|-----------|---------------|
| **Gì thay đổi** | P(X) — input distribution | P(Y\|X) — relationship input→output |
| **Ví dụ** | Users shift desktop → mobile (input features change) | COVID changes buying patterns (same features, different outcomes) |
| **Detect** | KS test, PSI on features | Monitor prediction accuracy vs ground truth |
| **Fix** | Retrain on new data distribution | Retrain + possibly new features/architecture |
| **Speed** | Usually gradual | Can be sudden (events) or gradual (trends) |

- **Follow-up**: "Feature drift vs label drift?" → Feature drift = subset of data drift (specific feature distributions shift). Label drift = class distribution changes (e.g., fraud rate increases).

### Q24: PSI (Population Stability Index)?
**A**: Measures distribution shift between reference (training) and production data.
- Formula: PSI = Σ (actual% - expected%) × ln(actual% / expected%)
- **PSI < 0.1**: stable ✅ → no action
- **0.1 - 0.2**: moderate ⚠️ → investigate
- **PSI > 0.2**: significant 🔴 → retrain
- Compute per feature → identify WHICH features drifted.

### Q25: Monitoring metrics cần track?
**A**: 4 categories:
1. **Model quality**: accuracy, F1, AUC (needs ground truth, often delayed)
2. **Prediction behavior**: prediction distribution, confidence distribution, output entropy
3. **System**: latency P50/P95/P99, error rate, throughput, GPU utilization
4. **Business**: revenue, conversion, user satisfaction (ultimate measure)
- **Key insight**: Model metrics lag (need labels). Proxy metrics (prediction distribution shifts) give early warning.

### Q26: Retraining triggers?
**A**: 
1. **Scheduled**: monthly/quarterly (simple, reliable)
2. **Performance drop**: accuracy below threshold (needs ground truth)
3. **Drift detected**: PSI > 0.2 on key features (no ground truth needed)
4. **Data volume**: accumulated N new labeled samples
5. **Event-driven**: business event (new product launch, market change)
- **Best practice**: combine scheduled (baseline) + drift-triggered (responsive).

---

## Cloud & Infrastructure (4 câu)

### Q27: Cloud platform nào cho ML?
**A**: 
- **GCP**: Best AI ecosystem (Vertex AI, TPUs, BigQuery ML). Google's own models.
- **AWS**: Biggest market (SageMaker end-to-end). Most third-party integrations.
- **Azure**: Microsoft enterprise integration. OpenAI exclusive partnership.
- **Decision**: depends on existing cloud infra + team expertise. All viable.
- **Follow-up**: "Serverless ML serving?" → Cloud Run (GCP), Lambda (AWS), Azure Functions. Pay-per-request, scale to zero. Cons: cold starts, no GPU usually.

### Q28: Spot/preemptible instances cho training?
**A**: 60-90% cheaper nhưng can be interrupted anytime.
- **Must**: checkpoint frequently (every epoch) → resume on new instance
- **Only for training** — never for serving (could lose mid-request)
- **Pattern**: request spot → if interrupted, auto-request new spot → resume from checkpoint

### Q29: GPU selection guide?
**A**: 

| Use case | GPU | Why |
|----------|-----|-----|
| Fine-tune LLM (7B) | A100 40GB / H100 | VRAM for model + optimizer |
| Inference serving | T4 / L4 | Cost-efficient, good INT8 |
| Training on budget | RTX 4090 (24GB) | Best consumer GPU |
| Development/testing | T4 (free Colab) | Good enough for prototyping |

### Q30: Cost optimization strategies?
**A**: 
1. **Spot training** (60-90% savings)
2. **Scale-to-zero serving** (Cloud Run, no cost at idle)
3. **Model compression** (INT8 quantization → smaller GPU → cheaper)
4. **Right-size instances** (don't use A100 for a 100MB model)
5. **Scheduled scaling** (scale down overnight/weekends)
6. **Cache inference results** (Redis, avoid duplicate LLM calls)

---

## LLMOps — ML Operations cho LLM Era (5 câu) 🆕

### Q31: LLMOps khác gì traditional MLOps?
**A**: 

| Aspect | Traditional MLOps | LLMOps |
|--------|------------------|--------|
| **Artifact** | Model weights (.pth) | Prompt template + model config |
| **Versioning** | Model version | Prompt version (hash of template) |
| **Evaluation** | Accuracy, F1 (automated) | LLM-as-Judge, human eval (expensive) |
| **Cost tracking** | GPU hours | Token usage × pricing per model |
| **A/B testing** | Model A vs Model B | Prompt A vs Prompt B (same model) |
| **Monitoring** | Feature drift | Prompt injection, hallucination rate, refusal rate |
| **Iteration speed** | Days (retrain) | Minutes (change prompt) |

- **Key insight**: LLMOps is faster iteration (prompt change = deploy in seconds) but harder evaluation (no single metric).

### Q32: Prompt versioning — cách quản lý prompts trong production?
**A**: 
```python
import hashlib

class PromptRegistry:
    def __init__(self):
        self.versions = {}
    
    def register(self, name: str, template: str, metadata: dict = None):
        version_hash = hashlib.sha256(template.encode()).hexdigest()[:8]
        self.versions[f"{name}:{version_hash}"] = {
            "template": template,
            "hash": version_hash,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }
        return version_hash

# Usage
registry = PromptRegistry()
v1 = registry.register("grading", "Score this essay: {essay}", 
                        {"model": "gpt-4o", "temperature": 0.3})
# Track: which prompt version → which outputs → which user feedback
```
- **Production pattern**: store prompt versions in DB, link to evaluation results, enable rollback.

### Q33: LLM cost tracking — cách tính chi phí?
**A**: 
```python
# Pricing table (2025-2026)
PRICING = {
    "gpt-4o":       {"input": 2.50, "output": 10.00},  # per 1M tokens
    "gpt-4o-mini":  {"input": 0.15, "output": 0.60},
    "claude-3.5":   {"input": 3.00, "output": 15.00},
    "gemini-2.0":   {"input": 0.075, "output": 0.30},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

# Example: 1000 requests, avg 500 input + 200 output tokens
daily_cost = 1000 * estimate_cost("gpt-4o", 500, 200)
# = 1000 × (500×2.50 + 200×10.00) / 1M = $3.25/day
```
- **Follow-up**: "Cách giảm cost?" → (1) Route simple queries to mini model, (2) Cache frequent responses, (3) Prompt compression, (4) Batch API calls.

### Q34: LLM evaluation pipeline?
**A**: Không có single metric — cần multi-dimensional evaluation:
1. **Automated**: BLEU/ROUGE (reference-based), BERTScore
2. **LLM-as-Judge**: GPT-4 evaluates outputs (scale 1-5 trên relevance, coherence, helpfulness)
3. **Human eval**: Periodic annotation (gold standard nhưng expensive)
4. **A/B testing**: prompt A vs B on real traffic → measure business metrics

**Key principle**: automated for regression testing (run on every change), human eval for periodic validation.

### Q35: LLM observability — LangSmith, Helicone?
**A**: Track **every LLM call** in production:
- **Input**: prompt, model, temperature, tokens
- **Output**: response, tokens, latency, cost
- **Trace**: multi-step chains (RAG: embed → search → rerank → generate)
- **Tools**: LangSmith (LangChain native), Helicone (proxy-based, any provider), Langfuse (open-source)
- **Why**: debug failures, optimize prompts, track cost, detect hallucinations.

---

## Feature Stores (3 câu) 🆕

### Q36: Feature store là gì? Khi nào cần?
**A**: Centralized repository cho ML features. Solves:
- **Consistency**: same feature logic for training AND serving (no training-serving skew)
- **Reuse**: team shares features across models (avoid duplicate computation)
- **Time travel**: get features AS-OF a specific timestamp (for reproducibility)

**Khi cần**: multiple models sharing features, real-time serving needs, team > 3 ML engineers.
**Khi KHÔNG cần**: single model, batch-only inference, small team.

### Q37: Feast architecture?
**A**: Open-source feature store. Components:
- **Offline store**: warehouse (BigQuery, Redshift) cho training data
- **Online store**: low-latency DB (Redis, DynamoDB) cho serving
- **Feature registry**: definitions + metadata (who created, data source, freshness)
- **Materialization**: scheduled job pushes features from offline → online store

### Q38: Training-serving skew — tại sao nguy hiểm?
**A**: Feature computed **differently** between training and serving → model performance drops silently.
- **Ví dụ**: training tính `user_avg_spending` trên 30 ngày data. Serving tính trên 7 ngày (do performance). → Model sees different feature distribution → wrong predictions.
- **Fix**: Feature store ensures SAME transformation code for both paths.
- **Follow-up**: "Cách detect skew?" → Log predictions + features in production, compare feature distributions vs training set regularly.
