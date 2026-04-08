# 📊 AI Engineering — Comparison Tables

> Side-by-side comparisons for interview prep. Khi interviewer hỏi "A vs B?" → mở file này.

---

## 1. ML Models

### Supervised Learning

| Model | Pros | Cons | Best For | Complexity |
|-------|------|------|----------|:----------:|
| **Linear Regression** | Simple, interpretable, fast | Linear only | Baseline, simple relationships | O(nd²) |
| **Logistic Regression** | Probabilistic, interpretable | Linear boundary | Binary classification, baseline | O(nd²) |
| **Decision Tree** | Interpretable, no scaling | Overfits easily | Explainability required | O(n·d·log n) |
| **Random Forest** | Robust, handles noise | Slow predict, memory | General tabular (balanced) | O(n·d·log n × T) |
| **XGBoost** | Best tabular accuracy | Tuning required, slow train | Kaggle / tabular production | O(n·d × T) |
| **LightGBM** | Fastest gradient boosting | Overfits small data | Large datasets | O(n·d × T) |
| **SVM** | Great margins, kernel trick | Slow on large data, no prob | Small datasets, text | O(n²·d) |
| **KNN** | No training, simple | Slow predict, curse of dim | Small data, recommendation | O(nd) per query |

### Unsupervised Learning

| Algorithm | Type | Pros | Cons | K Required? |
|-----------|------|------|------|:-----------:|
| **K-Means** | Partitional | Fast, scalable | Spherical clusters only | ✅ |
| **DBSCAN** | Density-based | Arbitrary shapes, handles noise | Sensitive to ε, MinPts | ❌ |
| **GMM** | Probabilistic | Soft assignments, elliptical | Sensitive to init, K needed | ✅ |
| **Hierarchical** | Agglomerative | No K needed, dendrogram | O(n³), not scalable | ❌ |
| **HDBSCAN** | Density-based | Variable density, robust | Slow on very large data | ❌ |

---

## 2. Deep Learning Architectures

### CNN Architectures

| Model | Params | Top-1 Acc | Speed | Best For |
|-------|:------:|:---------:|:-----:|----------|
| **ResNet-50** | 25M | 76.1% | ⚡⚡⚡ | Baseline, transfer learning |
| **EfficientNet-B4** | 19M | 83.0% | ⚡⚡ | Best accuracy/param ratio |
| **ViT-B/16** | 86M | 84.0% | ⚡⚡ | Large datasets, transformers |
| **ConvNeXt-T** | 29M | 82.1% | ⚡⚡⚡ | Modern CNN, ViT-competitive |
| **MobileNetV3** | 5.4M | 75.2% | ⚡⚡⚡⚡ | Mobile / edge deployment |
| **Swin-T** | 28M | 81.3% | ⚡⚡ | Object detection backbone |

### Segmentation Models

| Model | Backbone | mIoU (ADE20K) | Speed | Best For |
|-------|----------|:------------:|:-----:|----------|
| **U-Net** | Custom | ~70% | ⚡⚡⚡ | Medical imaging, small datasets |
| **DeepLabV3+** | ResNet-101 | ~46% | ⚡⚡ | General segmentation |
| **SegFormer-B5** | MiT-B5 | ~51% | ⚡⚡ | SOTA lightweight |
| **Mask2Former** | Swin-L | ~57% | ⚡ | Instance + panoptic |
| **SAM** | ViT-H | N/A | ⚡ | Zero-shot, interactive |

---

## 3. LLM Models

| Model | Provider | Context | Cost (1M tokens) | Best For |
|-------|----------|:-------:|:-----------------:|----------|
| **GPT-4o** | OpenAI | 128K | $2.5 / $10 | Complex reasoning, coding |
| **GPT-4o-mini** | OpenAI | 128K | $0.15 / $0.60 | Cost-efficient, simple tasks |
| **Claude 3.5 Sonnet** | Anthropic | 200K | $3 / $15 | Long context, coding |
| **Gemini 1.5 Pro** | Google | 2M | $1.25 / $5 | Massive context, multimodal |
| **Llama 3.1 70B** | Meta | 128K | Self-host | Open-source, fine-tuning |
| **Mistral Large** | Mistral | 128K | $2 / $6 | European, multilingual |

---

## 4. Vector Databases

| DB | Type | Scale | Speed | Best For |
|----|------|:-----:|:-----:|----------|
| **ChromaDB** | In-process | <1M | ⚡⚡⚡ | Prototyping, local dev |
| **pgvector** | Postgres ext | <10M | ⚡⚡ | Existing Postgres, Supabase |
| **Pinecone** | Cloud SaaS | 1B+ | ⚡⚡⚡ | Production, managed |
| **Qdrant** | Self-host/Cloud | 100M+ | ⚡⚡⚡ | Rust-fast, filtering |
| **Weaviate** | Self-host/Cloud | 100M+ | ⚡⚡ | Multi-modal, GraphQL |
| **Milvus** | Self-host | 1B+ | ⚡⚡⚡ | Large-scale, GPU support |

---

## 5. Embedding Models

| Model | Dim | MTEB Score | Speed | Language |
|-------|:---:|:---------:|:-----:|----------|
| **text-embedding-3-small** | 1536 | ~62% | ⚡⚡⚡ (API) | Multi |
| **text-embedding-3-large** | 3072 | ~65% | ⚡⚡ (API) | Multi |
| **all-MiniLM-L6-v2** | 384 | ~63% | ⚡⚡⚡ (local) | EN |
| **bge-large-en-v1.5** | 1024 | ~64% | ⚡⚡ (local) | EN |
| **jina-embeddings-v3** | 1024 | ~66% | ⚡⚡ (local) | Multi |
| **Cohere embed-v3** | 1024 | ~65% | ⚡⚡ (API) | Multi |

---

## 6. RAG Components

### Chunking Strategies

| Strategy | Speed | Quality | Variable Size | Context Preserved |
|----------|:-----:|:-------:|:----:|:----:|
| **Fixed-size** | ⚡⚡⚡ | ⭐ | ❌ | ❌ |
| **Recursive** | ⚡⚡⚡ | ⭐⭐ | Partial | ❌ |
| **Semantic** | ⚡ | ⭐⭐⭐ | ✅ | Partial |
| **Document-aware** | ⚡⚡ | ⭐⭐⭐ | ✅ | ✅ |
| **Late chunking** | ⚡ | ⭐⭐⭐⭐ | ✅ | ✅ |
| **Contextual (Anthropic)** | 🐢 | ⭐⭐⭐⭐ | ✅ | ✅ |

### Reranking Models

| Model | Type | Speed | Quality | Cost |
|-------|------|:-----:|:-------:|------|
| **MiniLM-L6** | Cross-encoder | ⚡⚡ | ⭐⭐ | Free (local) |
| **bge-reranker-v2-m3** | Cross-encoder | ⚡ | ⭐⭐⭐⭐ | Free (local) |
| **ColBERTv2** | Late interaction | ⚡⚡ | ⭐⭐⭐ | Free (local) |
| **Cohere Rerank v3** | API | ⚡⚡ | ⭐⭐⭐⭐ | $2/1K queries |
| **LLM reranking** | LLM | 🐢 | ⭐⭐⭐⭐⭐ | $$ per query |

---

## 7. Agent Frameworks

| Framework | Paradigm | Flexibility | Learning Curve | Best For |
|-----------|----------|:-----------:|:-----------:|----------|
| **LangGraph** | Graph-based | ⭐⭐⭐⭐⭐ | Moderate | Production agents, complex flows |
| **CrewAI** | Role-based | ⭐⭐⭐ | Easy | Content pipelines, team simulation |
| **AutoGen** | Conversation | ⭐⭐⭐⭐ | Moderate | Multi-agent chat, coding |
| **LangChain** | Chain-based | ⭐⭐⭐⭐ | Hard | RAG pipelines, tool orchestration |
| **Semantic Kernel** | Plugin-based | ⭐⭐⭐ | Moderate | Enterprise (.NET/Java) |

---

## 8. Deployment Platforms

| Platform | GPU | Auto-scale | Cold Start | Cost Model |
|----------|:---:|:----------:|:----------:|------------|
| **Vercel** | ❌ | ✅ Auto | ~100ms | Free → $20/mo |
| **Cloud Run** | ✅ L4 | ✅ Auto | 5-30s | Pay per use |
| **Modal** | ✅ A100 | ✅ Auto | 10-60s | Per second |
| **Railway** | ❌ | ✅ Auto | ~2s | $5/mo + usage |
| **K8s (GKE)** | ✅ Any | HPA | Depends | Variable |
| **AWS Lambda** | ❌ | ✅ Auto | 5-15s | Per request |

---

## 9. MLOps Tools

| Category | Tool | Type | Best For |
|----------|------|------|----------|
| **Experiment Tracking** | MLflow | Self-host | Open-source, full control |
| | Weights & Biases | Cloud | Visualization, collaboration |
| **Data Versioning** | DVC | CLI | Git-like data versioning |
| | LakeFS | Self-host | Data lake versioning |
| **Model Serving** | TorchServe | Self-host | PyTorch models |
| | Triton | Self-host | Multi-framework, GPU |
| | vLLM | Self-host | LLM serving, high throughput |
| **Monitoring** | Evidently | Library | Data/model drift |
| | Prometheus | Self-host | Infrastructure metrics |
| **Orchestration** | Prefect | Cloud/Self | Python-native workflows |
| | Airflow | Self-host | Complex DAG pipelines |

---

## 10. Fine-tuning Methods

| Method | Params Updated | Memory | Quality | Cost |
|--------|:-------------:|:------:|:-------:|:----:|
| **Full Fine-tune** | 100% | ⚡⚡⚡⚡ Very High | ⭐⭐⭐⭐⭐ | $$$ |
| **LoRA** | ~0.1-1% | ⚡⚡ Medium | ⭐⭐⭐⭐ | $ |
| **QLoRA** | ~0.1-1% | ⚡ Low | ⭐⭐⭐⭐ | $ |
| **Prefix Tuning** | ~0.1% | ⚡ Low | ⭐⭐⭐ | $ |
| **Prompt Tuning** | ~0.01% | ⚡ Minimal | ⭐⭐ | ¢ |
| **RLHF (PPO)** | ~1-10% | ⚡⚡⚡ High | ⭐⭐⭐⭐⭐ | $$$$ |
| **DPO** | ~1-10% | ⚡⚡ Medium | ⭐⭐⭐⭐ | $$ |

---

## 💡 Usage Tips

```
Interviewer: "Compare X and Y"
Your answer structure:

1. "Both solve [problem], but differ in [key aspect]."
2. "X is better when [conditions]. Y is better when [conditions]."
3. "In practice, I'd choose [recommendation] because [reason]."
4. "Trade-off: X gives [advantage] but costs [disadvantage]."
```
