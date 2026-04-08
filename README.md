# 🚀 AI Engineering — Complete Learning Workspace

> **Hệ thống hoá toàn bộ kiến thức AI Engineering** — Từ nền tảng → Production
>
> 110+ files · 62 deep-dive docs (expert-level) · 24 runnable examples · 600+ interview Q&A
>
> *Cập nhật: April 2026 — Wave A/B/C/D Complete*

### 📋 Quick References (NEW!)
| | File | Mô tả |
|---|------|-------|
| 📋 | [**CHEATSHEET.md**](./CHEATSHEET.md) | One-page-per-topic quick reference — mở trước interview |
| 📊 | [**COMPARISON_TABLES.md**](./COMPARISON_TABLES.md) | Side-by-side comparisons (Models, DBs, Frameworks, Tools) |

---

## 🗺️ Learning Path

```
Level 0             Level 1                Level 2              Level 3
─────────           ──────────             ──────────           ──────────
Foundations    →    Classical ML     →     AI Agents      →    Advanced RAG
Python, Math        Sklearn, XGBoost       LangGraph, MCP       Hybrid Search, GraphRAG

                    Deep Learning    →     Speech AI      →    Full-Stack AI
                    PyTorch, CNN           Whisper, TTS         FastAPI, React

                    NLP & LLM        →     MLOps          →    Interview Prep
                    Transformers, RAG      MLflow, Docker       335+ Questions
```

---

## 📂 Cấu trúc Workspace

| # | Folder | Lĩnh vực | Docs | Examples | Interview |
|---|--------|----------|------|----------|-----------|
| 00 | [`00_foundations/`](./00_foundations/) | Python, Math, Git, Docker, API, SQL, Data Eng | 7 | 2 | 43 Q&A |
| 01 | [`01_classical_ml/`](./01_classical_ml/) | Supervised, Unsupervised, Feature Eng, Evaluation | 6 | 2 | 35 Q&A |
| 02 | [`02_deep_learning_cv/`](./02_deep_learning_cv/) | Neural Nets, CNN, Transfer Learning, Segmentation | 6 | 2 | 30 Q&A |
| 03 | [`03_nlp_llm/`](./03_nlp_llm/) | Transformers, LLM, Prompts, RAG, Fine-tuning | 7 | 3 | 35 Q&A |
| 04 | [`04_mlops/`](./04_mlops/) | MLflow, DVC, Docker, CI/CD, Monitoring, Cloud | 6 | 3 | 30 Q&A |
| 05 | [`05_ai_agents/`](./05_ai_agents/) | LangGraph, MCP, Memory, Multi-Agent, Safety | 6 | 3 | 30 Q&A |
| 06 | [`06_speech_ai/`](./06_speech_ai/) | Audio, ASR/Whisper, TTS, Diarization, Voice Agents | 6 | 3 | 26 Q&A |
| 07 | [`07_advanced_rag/`](./07_advanced_rag/) | Chunking, Hybrid Search, Reranking, Graph RAG | 6 | 3 | 25 Q&A |
| 08 | [`08_fullstack_ai/`](./08_fullstack_ai/) | FastAPI, Streaming, Auth, React, Deployment | 6 | 3 | 25 Q&A |
| 09 | [`09_interview_prep/`](./09_interview_prep/) | Tổng hợp 335+ câu Technical + Behavioral + Coding | — | — | 335+ Q&A |
| | | **TỔNG** | **62** | **24** | **≈600+** |

---

## 📚 MỤC LỤC CHI TIẾT

### 🔹 00 — Foundations (Nền tảng)
| Doc | Chủ đề | Highlights |
|-----|--------|------------|
| [00_python_basics](./00_foundations/docs/00_python_basics.md) | Python cơ bản | Data Types, OOP, Control Flow, File I/O, Pitfalls |
| [01_python_advanced](./00_foundations/docs/01_python_advanced.md) | Python nâng cao | Decorator, Generator, Async/Await, Type Hints |
| [02_math_for_ai](./00_foundations/docs/02_math_for_ai.md) | Toán cho AI | Linear Algebra, Probability, Calculus, Optimization |
| [03_git_workflow](./00_foundations/docs/03_git_workflow.md) | Git workflow | Branching, Conventional Commits, CI/CD |
| [04_docker_essentials](./00_foundations/docs/04_docker_essentials.md) | Docker | Dockerfile, Multi-stage, Compose, Volumes |
| [05_api_design](./00_foundations/docs/05_api_design.md) | REST API | FastAPI, Pydantic, HTTP methods, Auth |
| [06_sql_essentials](./00_foundations/docs/06_sql_essentials.md) | SQL | JOIN, Subquery, Window Functions, Index |
| [07_data_engineering](./00_foundations/docs/07_data_engineering.md) | Data Engineering | Spark, Kafka, ETL/ELT, Data Pipelines |
| 💻 Examples | | `python_async_demo.py` · `fastapi_demo.py` |

---

### 🔹 01 — Classical ML
| Doc | Chủ đề | Highlights |
|-----|--------|------------|
| [01_supervised_learning](./01_classical_ml/docs/01_supervised_learning.md) | Supervised | Linear/Logistic Regression, SVM, Trees, Ensemble |
| [02_unsupervised_learning](./01_classical_ml/docs/02_unsupervised_learning.md) | Unsupervised | K-Means, DBSCAN, PCA, t-SNE |
| [03_feature_engineering](./01_classical_ml/docs/03_feature_engineering.md) | Feature Eng | Encoding, Scaling, Selection, Missing Values |
| [04_evaluation_metrics](./01_classical_ml/docs/04_evaluation_metrics.md) | Evaluation | Accuracy, F1, AUC-ROC, Confusion Matrix |
| [05_hyperparameter_tuning](./01_classical_ml/docs/05_hyperparameter_tuning.md) | Tuning | Grid/Random Search, Optuna, Cross-Validation |
| [06_practical_pipeline](./01_classical_ml/docs/06_practical_pipeline.md) | Pipeline | Sklearn Pipeline, Column Transformer, MLflow |
| 💻 Examples | | `sklearn_pipeline.py` · `optuna_tuning.py` |

---

### 🔹 02 — Deep Learning & Computer Vision
| Doc | Chủ đề | Highlights |
|-----|--------|------------|
| [01_neural_networks](./02_deep_learning_cv/docs/01_neural_networks.md) | Neural Nets | Backprop, Activation, Optimizer, Regularization |
| [02_cnn_architectures](./02_deep_learning_cv/docs/02_cnn_architectures.md) | CNN | ResNet, EfficientNet, ViT, MobileNet |
| [03_transfer_learning](./02_deep_learning_cv/docs/03_transfer_learning.md) | Transfer Learn | Pretrained, Fine-tuning, Feature Extraction |
| [04_segmentation_detection](./02_deep_learning_cv/docs/04_segmentation_detection.md) | Seg & Detect | U-Net, YOLO, Mask R-CNN, mIoU |
| [05_training_recipes](./02_deep_learning_cv/docs/05_training_recipes.md) | Training | LR Schedule, Mixed Precision, Data Aug |
| [06_model_deployment](./02_deep_learning_cv/docs/06_model_deployment.md) | Deploy | ONNX, TorchScript, TensorRT, Quantization |
| 💻 Examples | | `pytorch_basics.py` · `cnn_classifier.py` |

---

### 🔹 03 — NLP & LLM
| Doc | Chủ đề | Highlights |
|-----|--------|------------|
| [01_transformers](./03_nlp_llm/docs/01_transformers.md) | Transformers | Attention, BERT, GPT, Positional Encoding |
| [02_llm_fundamentals](./03_nlp_llm/docs/02_llm_fundamentals.md) | LLM | Scaling Laws, RLHF, Tokenization, Context |
| [03_prompt_engineering](./03_nlp_llm/docs/03_prompt_engineering.md) | Prompting | CoT, Few-shot, System Prompt, Output Format |
| [04_rag_architecture](./03_nlp_llm/docs/04_rag_architecture.md) | RAG | Embedding, Vector DB, Chunking, Pipeline |
| [05_finetuning](./03_nlp_llm/docs/05_finetuning.md) | Fine-tuning | LoRA, QLoRA, PEFT, Dataset Preparation |
| [06_evaluation](./03_nlp_llm/docs/06_evaluation.md) | Evaluation | BLEU, ROUGE, LLM-as-Judge, Human Eval |
| [07_vector_databases](./03_nlp_llm/docs/07_vector_databases.md) | Vector DB | ChromaDB, Pinecone, pgvector, HNSW |
| 💻 Examples | | `openai_basics.py` · `embeddings_demo.py` · `rag_pipeline.py` |

---

### 🔹 04 — MLOps
| Doc | Chủ đề | Highlights |
|-----|--------|------------|
| [01_experiment_tracking](./04_mlops/docs/01_experiment_tracking.md) | Tracking | MLflow, W&B, Autolog, Model Registry |
| [02_data_versioning](./04_mlops/docs/02_data_versioning.md) | DVC | Data Versioning, Pipelines, Experiments |
| [03_containerization](./04_mlops/docs/03_containerization.md) | Docker/K8s | Multi-stage Build, Compose, Kubernetes |
| [04_cicd_ml](./04_mlops/docs/04_cicd_ml.md) | CI/CD | GitHub Actions, Quality Gates, Canary |
| [05_model_monitoring](./04_mlops/docs/05_model_monitoring.md) | Monitoring | Drift Detection, Alerting, Retraining |
| [06_cloud_platforms](./04_mlops/docs/06_cloud_platforms.md) | Cloud | GCP Vertex AI, AWS SageMaker, Azure ML |
| 💻 Examples | | `mlflow_tracking.py` · `dvc_pipeline.py` · `monitoring_dashboard.py` |

---

### 🔹 05 — AI Agents
| Doc | Chủ đề | Highlights |
|-----|--------|------------|
| [01_agent_fundamentals](./05_ai_agents/docs/01_agent_fundamentals.md) | Fundamentals | ReAct, Tool Use, Reasoning Loop |
| [02_langgraph](./05_ai_agents/docs/02_langgraph.md) | LangGraph | State Graph, Conditional Edges, Checkpointing |
| [03_tool_design](./05_ai_agents/docs/03_tool_design.md) | MCP & Tools | Model Context Protocol, Tool Design |
| [04_memory_systems](./05_ai_agents/docs/04_memory_systems.md) | Memory | Short/Long-term, RAG Memory, Summarization |
| [05_multi_agent](./05_ai_agents/docs/05_multi_agent.md) | Multi-Agent | CrewAI, Supervisor, Debate Pattern |
| [06_agent_safety](./05_ai_agents/docs/06_agent_safety.md) | Safety | Guardrails, Human-in-the-loop, Sandboxing |
| 💻 Examples | | `react_agent.py` · `langgraph_workflow.py` · `mcp_tool_server.py` |

---

### 🔹 06 — Speech AI
| Doc | Chủ đề | Highlights |
|-----|--------|------------|
| [01_audio_fundamentals](./06_speech_ai/docs/01_audio_fundamentals.md) | Audio | Sampling, Spectrogram, MFCC, Preprocessing |
| [02_asr_speech_to_text](./06_speech_ai/docs/02_asr_speech_to_text.md) | ASR | Whisper, faster-whisper, Streaming, WER |
| [03_tts_text_to_speech](./06_speech_ai/docs/03_tts_text_to_speech.md) | TTS | F5-TTS, Bark, XTTS, Voice Cloning |
| [04_speaker_diarization](./06_speech_ai/docs/04_speaker_diarization.md) | Diarization | Pyannote, Speaker Embeddings, DER |
| [05_voice_agents](./06_speech_ai/docs/05_voice_agents.md) | Voice Agent | Real-time Pipeline, Latency, Barge-in |
| [06_production_speech](./06_speech_ai/docs/06_production_speech.md) | Production | Noise Handling, Monitoring, Cost |
| 💻 Examples | | `audio_processing.py` · `whisper_demo.py` · `tts_demo.py` |

---

### 🔹 07 — Advanced RAG
| Doc | Chủ đề | Highlights |
|-----|--------|------------|
| [01_chunking_strategies](./07_advanced_rag/docs/01_chunking_strategies.md) | Chunking | Semantic, Recursive, Parent-Child, Agentic |
| [02_hybrid_search](./07_advanced_rag/docs/02_hybrid_search.md) | Hybrid Search | BM25 + Vector, RRF Fusion, HyDE |
| [03_reranking](./07_advanced_rag/docs/03_reranking.md) | Reranking | Cross-encoder, Cohere Rerank, ColBERT |
| [04_multimodal_rag](./07_advanced_rag/docs/04_multimodal_rag.md) | Multi-modal | PDF Parsing, Vision RAG, Table RAG |
| [05_graph_rag](./07_advanced_rag/docs/05_graph_rag.md) | Graph RAG | Knowledge Graphs, Neo4j, Microsoft GraphRAG |
| [06_rag_evaluation](./07_advanced_rag/docs/06_rag_evaluation.md) | Evaluation | RAGAS, LLM-as-Judge, Regression Testing |
| 💻 Examples | | `chunking_demo.py` · `hybrid_search.py` · `rag_evaluation.py` |

---

### 🔹 08 — Full-Stack AI
| Doc | Chủ đề | Highlights |
|-----|--------|------------|
| [01_api_design](./08_fullstack_ai/docs/01_api_design.md) | API Design | FastAPI, Pydantic, Middleware, Error Handling |
| [02_streaming_sse](./08_fullstack_ai/docs/02_streaming_sse.md) | Streaming | SSE, WebSocket, Token Streaming |
| [03_auth_security](./08_fullstack_ai/docs/03_auth_security.md) | Auth | JWT, API Keys, Rate Limiting, Prompt Injection |
| [04_frontend_ai](./08_fullstack_ai/docs/04_frontend_ai.md) | Frontend | React Chat UI, Next.js, Vercel AI SDK |
| [05_state_management](./08_fullstack_ai/docs/05_state_management.md) | State | Zustand, Chat History, Optimistic Updates |
| [06_deployment](./08_fullstack_ai/docs/06_deployment.md) | Deployment | Vercel, Cloud Run, Docker, CI/CD |
| 💻 Examples | | `fastapi_server.py` · `websocket_chat.py` · `auth_demo.py` |

---

### 🔹 09 — Interview Prep
| File | Nội dung | Số câu |
|------|----------|--------|
| [01_technical_questions](./09_interview_prep/) | Technical Q&A tổng hợp | 100+ |
| [02_behavioral](./09_interview_prep/) | Behavioral / STAR method | 20 |
| [03_coding_challenges](./09_interview_prep/) | Coding exercises | 15 |
| [04_system_design](./09_interview_prep/) | System Design (ML focus) | 5 |
| [05_portfolio_guide](./09_interview_prep/) | Portfolio & Resume tips | — |

---

## 🛠️ Setup

```bash
# 1. Clone & setup virtual environment
cd "d:\Code\New folder"
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 2. Install core dependencies
pip install -r requirements.txt
```

## 📖 Cách sử dụng

### 🎯 Lộ trình học
```
Tuần 1-2:  00_foundations      ← Python nâng cao, Math, Git, Docker
Tuần 3-4:  01_classical_ml    ← Sklearn, Feature Engineering
           02_deep_learning_cv ← PyTorch, CNN, Transfer Learning
Tuần 5-6:  03_nlp_llm         ← Transformer, LLM, RAG
           04_mlops           ← MLflow, Docker, CI/CD
Tuần 7-8:  05_ai_agents       ← LangGraph, MCP, Multi-Agent
           06_speech_ai       ← Whisper, TTS, Voice Agents
Tuần 9-10: 07_advanced_rag    ← Hybrid Search, Graph RAG
           08_fullstack_ai    ← FastAPI, React, Deployment
Tuần 11+:  09_interview_prep  ← Tổng ôn 335+ câu hỏi
```

### 📝 Chu trình mỗi module
1. **📖 Đọc `docs/`** — Lý thuyết chuyên sâu (song ngữ Việt-Anh)
2. **💻 Chạy `examples/`** — Code minh họa chạy được ngay (`python <file>.py`)
3. **✅ Ôn `interview/`** — Q&A với gợi ý trả lời
4. **🔄 Repeat** — Đọc lại phần chưa vững

### 🏃 Quick Start — Chạy thử ngay
```bash
# Chunking strategies comparison
python 07_advanced_rag/examples/chunking_demo.py

# Hybrid BM25 + Vector search
python 07_advanced_rag/examples/hybrid_search.py

# RAG evaluation with RAGAS metrics
python 07_advanced_rag/examples/rag_evaluation.py

# FastAPI demo server
python 08_fullstack_ai/examples/fastapi_server.py

# JWT + API Key authentication
python 08_fullstack_ai/examples/auth_demo.py
```

---

## 📊 Thống kê Workspace

| Metric | Số lượng |
|--------|----------|
| Tổng files | **110+** |
| Documentation | 62 deep-dive docs (expert-level) |
| Code examples | 24 runnable Python files |
| Interview Q&A (per-module) | 350+ câu |
| Interview Q&A (tổng ôn 09) | 335+ câu |
| Quick reference | CHEATSHEET + COMPARISON_TABLES |
| **Tổng Q&A** | **≈700+** |
| Modules | 10 (00 → 09) |
| Không cần API key | ✅ Tất cả examples chạy offline |

---

## 🔑 Key Technologies

```
Languages:     Python 3.11+ · TypeScript · SQL
ML/DL:         PyTorch · Sklearn · XGBoost · Optuna
LLM:           OpenAI API · LangChain · LangGraph
RAG:           ChromaDB · Pinecone · BM25 · RAGAS
Agents:        LangGraph · MCP · CrewAI
Speech:        Whisper · Pyannote · Bark · Librosa
MLOps:         MLflow · DVC · Docker · GitHub Actions
Cloud:         GCP Vertex AI · AWS SageMaker · Azure ML
Backend:       FastAPI · Uvicorn · Pydantic · JWT
Frontend:      React · Next.js · Vercel AI SDK · Zustand
```

---

<p align="center">
  <b>Made with ❤️ for AI Engineering Interview Preparation</b><br>
  <i>From zero to production-ready AI Engineer</i>
</p>
