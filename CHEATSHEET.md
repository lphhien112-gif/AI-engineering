# 📋 AI Engineering — Quick Reference Cheatsheet

> One-page-per-topic cheat sheets. Mở file này khi cần ôn nhanh trước interview.

---

## 🐍 Python Advanced

```python
# Decorator
def retry(n=3):
    def decorator(fn):
        def wrapper(*a, **kw):
            for i in range(n):
                try: return fn(*a, **kw)
                except Exception as e:
                    if i == n-1: raise
        return wrapper
    return decorator

# Generator (lazy, memory-efficient)
def read_large_file(path):
    with open(path) as f:
        for line in f:
            yield line.strip()

# Async
async def fetch_all(urls):
    async with aiohttp.ClientSession() as s:
        tasks = [s.get(u) for u in urls]
        return await asyncio.gather(*tasks)

# Context Manager
from contextlib import contextmanager
@contextmanager
def timer():
    t = time.time()
    yield
    print(f"{time.time()-t:.2f}s")

# Type Hints
def process(items: list[str], limit: int = 10) -> dict[str, float]: ...
```

---

## 📊 Classical ML

```
Model Selection:
  Tabular data → XGBoost / LightGBM (first try)
  Small data (<1K) → Logistic Regression / SVM  
  Interpretability → Decision Tree / Linear Models
  Anomaly detection → Isolation Forest / LOF

Metrics:
  Classification: Accuracy (balanced), F1 (imbalanced), AUC-ROC (ranking)
  Regression: RMSE (penalizes outliers), MAE (robust), R² (explained variance)
  Clustering: Silhouette [-1,1], Calinski-Harabasz (higher=better)

Feature Engineering:
  Missing → median (numerical), mode (categorical), flag column
  Categorical → Target encoding (high cardinality), One-hot (low)
  Numerical → StandardScaler (linear), none (tree-based)
  Text → TF-IDF (simple), sentence embeddings (semantic)
```

---

## 🧠 Deep Learning

```
Architecture Quick-Pick:
  Image classification → EfficientNet / ViT
  Object detection → YOLOv8 / RT-DETR
  Segmentation → SegFormer (MiT-B5) / U-Net++
  NLP → BERT (understanding) / GPT (generation)

Training Checklist:
  □ Learning rate: 1e-4 (fine-tune), 1e-3 (from scratch)
  □ Optimizer: AdamW (default), SGD+momentum (vision)
  □ Scheduler: CosineAnnealing + warmup (5-10% steps)
  □ Batch size: max that fits GPU → use gradient accumulation
  □ Mixed precision: fp16 (2x speed, 0.5x memory)
  □ EarlyStopping: patience=10, monitor=val_loss
  □ Data augmentation: ALWAYS for vision (Albumentations)

GPU Memory Tricks:
  1. torch.cuda.empty_cache()
  2. Gradient accumulation (simulate larger batch)
  3. Mixed precision (fp16)
  4. Gradient checkpointing (trade compute for memory)
  5. Smaller model / reduce batch size
```

---

## 🤖 LLM & Prompting

```
Prompt Templates:
  Zero-shot:   "Classify: {text} → positive/negative"
  Few-shot:    "Examples: ... Now classify: {text}"
  Chain-of-Thought: "Think step by step. {question}"
  System:      "You are a {role} who {constraints}."

Token Estimation:  1 token ≈ 4 chars (EN) ≈ 1-2 chars (VN)
Cost:  GPT-4o: $2.5/$10 per 1M tokens (in/out)
       GPT-4o-mini: $0.15/$0.60 per 1M (10x cheaper!)

RAG Pipeline:
  Query → [Embed] → [Vector Search top-50] → [Rerank top-5] → [LLM + Context] → Answer

Fine-tuning Decision:
  Prompt engineering → few-shot → RAG → fine-tuning (LoRA) → full fine-tuning
  Each step: more effort, more control, more cost
```

---

## 🕸️ RAG Advanced

```
Chunking: Recursive (default) → Semantic (better) → Late chunking (SOTA)
  Size: Q&A=500-1000 tokens, Summarization=1500-3000
  Overlap: 10-20% of chunk size

Search: Vector + BM25 (hybrid) → RRF fusion → Rerank (cross-encoder)
  Retrieve 30-50, rerank to top 3-5

Evaluation (RAGAS):
  Retrieval: Precision@K, Recall, MRR, NDCG, Hit Rate
  Generation: Faithfulness (>0.9), Answer Relevancy (>0.85)
  
GraphRAG: multi-hop reasoning, entity relationships, global themes
Multimodal: PDF→Unstructured/Docling, Images→GPT-4o description, Tables→NL+markdown
```

---

## 🤝 AI Agents

```
Patterns:
  ReAct:      Reason → Act → Observe → repeat
  Supervisor: Routes tasks to specialist agents
  Pipeline:   Agent1 → Agent2 → Agent3 (sequential)
  Debate:     AgentA ↔ AgentB → Judge → answer
  Map-Reduce: Parallel process → combine

Frameworks:
  LangGraph = flexible, graph-based, production
  CrewAI = simple, role-based, content pipelines
  
Memory:
  Short-term: conversation buffer (last N messages)
  Long-term: vector DB (semantic search past conversations)
  Summary: LLM summarizes old messages (compress)

Safety: Input guard → LLM check → Output guard → Human approval (risky actions)
```

---

## 🎤 Speech AI

```
ASR: Whisper (OpenAI) → faster-whisper (4x speed, CTranslate2)
TTS: F5-TTS (fast), Bark (multilingual), XTTS (voice cloning)
Diarization: Pyannote (who spoke when)

Key Metrics:
  ASR: WER (Word Error Rate) ← lower is better
  TTS: MOS (Mean Opinion Score 1-5) ← higher is better
  Diarization: DER (Diarization Error Rate) ← lower is better

Pipeline: Audio → VAD → ASR → Diarization → LLM → TTS → Audio
Latency target: <500ms TTFT for voice agents
```

---

## 🏗️ Full-Stack AI

```
Architecture:
  Frontend: Next.js → Vercel (free)
  Backend:  FastAPI → Cloud Run (auto-scale)
  Database: Supabase (Postgres + Auth + pgvector)
  Cache:    Redis (repeated queries, 80%+ hit rate)
  LLM:      OpenAI API / self-hosted vLLM

Streaming: SSE (90% of AI apps, including ChatGPT/Claude)
  Backend: async generator → "data: {json}\n\n"
  Frontend: fetch + ReadableStream reader

Auth: JWT (users) + API Key (services)
Rate limit: Tiered (free=10rpm, premium=300rpm)
Security: sanitize input → system prompt guard → output filter → content moderation
```

---

## 🚀 MLOps

```
Experiment Tracking: MLflow (self-hosted) / W&B (cloud)
Data Versioning: DVC (+ S3/GCS remote)
CI/CD: GitHub Actions → lint → test → build → deploy
Monitoring: Evidently (data drift), Prometheus+Grafana (infra)

Deployment:
  Docker → multi-stage, non-root, health check
  Cloud Run → min-instances=1, timeout=300s
  Model serving → ONNX (2-5x faster), TensorRT (GPU)

Key principle: 
  Code → Git | Data → DVC | Models → MLflow | Infra → Docker
```

---

## 💡 Interview Quick Tips

```
1. Always mention TRADE-OFFS (no "best" solution)
2. Start with SIMPLE approach → upgrade if needed
3. Production ≠ Research (monitoring, error handling, scaling)
4. Numbers matter: "reduces latency by 40%" > "makes it faster"
5. Ask clarifying questions before answering
6. Framework: Problem → Approach → Trade-offs → My recommendation
```
