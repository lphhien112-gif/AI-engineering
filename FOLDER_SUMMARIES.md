# 📋 Tổng Kết Kiến Thức — AI Engineering Knowledge Base

> **Cập nhật**: 2026-04-08
> **Tổng**: 10 modules | 83 files | ~927 KB content
> **Mục tiêu**: Expert-level, interview-ready, production-grade knowledge

---

## 00_foundations — Nền Tảng Lập Trình & Toán

| File | Size | Chủ đề chính |
|------|:----:|-------------|
| `00_python_basics.md` | 24 KB | Data types, control flow, functions, OOP, comprehensions |
| `01_python_advanced.md` | 25.5 KB | Decorators, generators, async/await, metaclasses, typing |
| `02_math_for_ai.md` | 21.5 KB | Linear Algebra, Probability, Calculus, Optimizers, **Attention Math**, **LR Scheduling** |
| `03_git_workflow.md` | 11 KB | Git branching, merge strategies, conventional commits |
| `04_docker_essentials.md` | 12.3 KB | Dockerfile, multi-stage builds, docker-compose, layer caching |
| `05_api_design.md` | 15.9 KB | REST, FastAPI, request/response patterns, error handling |
| `06_sql_essentials.md` | 13.6 KB | SQL queries, JOINs, window functions, indexing, CTEs |
| `07_data_engineering.md` | 17.9 KB | ETL, Pandas, Spark basics, data quality, batch vs streaming |
| `foundations_qa.md` | 19 KB | Q&A ôn tập tổng hợp |

**Điểm mạnh**: Rất dày kiến thức (164 KB), bao phủ đầy đủ stack. Attention Math có numerical example step-by-step. LR Scheduling có production code (Warmup+Cosine, OneCycleLR).

**Kiến thức nổi bật**:
- Attention Math: Tính toán Q×K^T/√d_k từng bước với 4 tokens
- Optimizer Math: Adam vs AdamW vs LAMB comparison
- √d_k scaling: Giải thích tại sao cần scale (interview answer ready)

---

## 01_classical_ml — Machine Learning Cổ Điển

| File | Size | Chủ đề chính |
|------|:----:|-------------|
| `01_supervised_learning.md` | 12.4 KB | Linear/Logistic Regression, Decision Trees, Ensemble, SVM, KNN, **Gradient Boosting Deep Dive**, **Optuna** |
| `02_unsupervised_learning.md` | 13.1 KB | K-Means, DBSCAN, PCA, t-SNE, UMAP |
| `03_feature_engineering.md` | 10.1 KB | Encoding, scaling, feature selection, feature importance |
| `04_evaluation_metrics.md` | 13.1 KB | Confusion matrix, ROC-AUC, regression metrics, segmentation IoU, **Calibration** |
| `05_hyperparameter_tuning.md` | 10.6 KB | Grid/Random/Bayesian search, Optuna, cross-validation |
| `06_practical_pipeline.md` | 13.3 KB | End-to-end ML pipeline, sklearn Pipeline, feature stores |
| `classical_ml_qa.md` | 13.7 KB | Q&A ôn tập |

**Điểm mạnh**: Gradient Boosting có giải thích residual learning step-by-step. XGBoost vs LightGBM vs CatBoost comparison table. Calibration section production-ready.

**Kiến thức nổi bật**:
- Gradient Boosting: F(x) = F₀ + η×h₁ + η×h₂ + ... (residual learning)
- XGBoost vs LightGBM vs CatBoost: Level-wise vs Leaf-wise vs Symmetric
- Probability Calibration: Brier score + Reliability diagram + CalibratedClassifierCV

---

## 02_deep_learning_cv — Deep Learning & Computer Vision

| File | Size | Chủ đề chính |
|------|:----:|-------------|
| `01_neural_networks.md` | 14.3 KB | Perceptron, backprop, activation functions, regularization |
| `02_cnn_architectures.md` | 10.3 KB | LeNet → ResNet → EfficientNet evolution |
| `03_transfer_learning.md` | 12.2 KB | Pre-trained models, fine-tuning strategies, feature extraction |
| `04_segmentation_detection.md` | 15.6 KB | U-Net, YOLO, Mask R-CNN, mAP metrics |
| `05_training_recipes.md` | 14.5 KB | Data augmentation, learning rate, mixed precision |
| `06_model_deployment.md` | 12.2 KB | ONNX, TensorRT, quantization, edge deployment |
| `deep_learning_qa.md` | 14.8 KB | Q&A ôn tập |

**Điểm mạnh**: Architecture evolution rõ ràng (CNN family tree). Training recipes practical với production settings. Deployment pipeline đầy đủ từ ONNX → TensorRT → Edge.

**Kiến thức nổi bật**:
- ResNet skip connections: Giải quyết vanishing gradient cho deep networks
- U-Net: Encoder-decoder + skip connections cho semantic segmentation
- Quantization: FP32 → INT8 (4x smaller, 2-4x faster, <1% accuracy loss)

---

## 03_nlp_llm — NLP & Large Language Models

| File | Size | Chủ đề chính |
|------|:----:|-------------|
| `01_transformers.md` | 15.1 KB | Self-attention, positional encoding, encoder/decoder, **Flash Attention** |
| `02_llm_fundamentals.md` | 16.3 KB | GPT architecture, tokenization, generation strategies |
| `03_prompt_engineering.md` | 15.6 KB | Zero/Few-shot, CoT, **Structured Output (JSON Schema, Instructor, Function Calling)** |
| `04_rag_architecture.md` | 14.5 KB | RAG pipeline, embedding, retrieval, generation |
| `05_finetuning.md` | 17.4 KB | LoRA/QLoRA, data preparation, **A/B Evaluation, LLM-as-Judge** |
| `06_evaluation.md` | 13.5 KB | BLEU, ROUGE, BERTScore, human evaluation |
| `07_vector_databases.md` | 12.5 KB | Qdrant, Pinecone, embedding models, ANN algorithms |
| `nlp_llm_qa.md` | 15.4 KB | Q&A ôn tập |

**Điểm mạnh**: Module lớn nhất (121.5 KB). Flash Attention giải thích từ HBM/SRAM tiling đến PyTorch SDPA code. Structured Output bao gồm 3 approach production-grade. Fine-tuning evaluation có LLM-as-Judge với blind comparison.

**Kiến thức nổi bật**:
- Flash Attention: O(N) memory thay O(N²), enable 128K+ context
- Structured Output: JSON Schema → Instructor → Function Calling comparison
- LoRA: rank=16, alpha=32, ~0.1% trainable params, fine-tune trên consumer GPU

---

## 04_mlops — ML Operations & Production

| File | Size | Chủ đề chính |
|------|:----:|-------------|
| `01_experiment_tracking.md` | 15.4 KB | MLflow, W&B, **LLM experiment tracking, cost tracking, prompt A/B** |
| `02_data_versioning.md` | 9.5 KB | DVC, data pipelines, versioning strategies |
| `03_containerization.md` | 10.4 KB | Docker for ML, GPU containers, multi-stage builds |
| `04_cicd_ml.md` | 13.7 KB | GitHub Actions, testing, automated training |
| `05_model_monitoring.md` | 11.8 KB | Data drift, model drift, alerting, Evidently AI |
| `06_cloud_platforms.md` | 10.7 KB | AWS/GCP/Azure ML services comparison |
| `mlops_qa.md` | 5.9 KB | Q&A ôn tập |

**Điểm mạnh**: LLM tracking section mới với prompt versioning (hash), cost tracking (pricing table), và LangSmith integration. CI/CD pipeline đầy đủ cho ML.

**Kiến thức nổi bật**:
- Experiment tracking: MLflow vs W&B comparison
- LLM cost: GPT-4o $2.50/M input, $10/M output (2025 prices)
- Model monitoring: Data drift detection + alerting pipeline

---

## 05_ai_agents — AI Agents & Tool Use

| File | Size | Chủ đề chính |
|------|:----:|-------------|
| `01_agent_fundamentals.md` | 14.7 KB | ReAct pattern, tool calling, framework comparison, **Agent Evaluation** |
| `02_langgraph.md` | 9.3 KB | LangGraph state machines, nodes, edges, checkpointing |
| `03_tool_design.md` | 11.7 KB | Tool schema, error handling, retry strategies |
| `04_memory_systems.md` | 12.4 KB | Short/long-term memory, conversation buffer, vector memory |
| `05_multi_agent.md` | 13.9 KB | Multi-agent orchestration, supervisor pattern, CrewAI |
| `06_agent_safety.md` | 11.1 KB | Guardrails, prompt injection defense, output filtering |
| `agents_qa.md` | 6.6 KB | Q&A ôn tập |

**Điểm mạnh**: Agent Evaluation framework mới với dataclass-based benchmark suite. Safety section comprehensive. Multi-agent pattern đa dạng.

**Kiến thức nổi bật**:
- ReAct: Reasoning + Acting loop (Thought → Action → Observation)
- Agent Evaluation: task_completion_rate, tool_accuracy, avg_steps metrics
- Safety: Prompt injection defense layers, output filtering

---

## 06_speech_ai — Speech AI & Voice

| File | Size | Chủ đề chính |
|------|:----:|-------------|
| `01_audio_fundamentals.md` | 10.1 KB | Sampling, spectrogram, MFCC, **SpecAugment, Noise Injection, Mel Visualization** |
| `02_asr_speech_to_text.md` | 10.2 KB | Whisper, Faster-Whisper, **VAD + Streaming Pipeline, WER Breakdown** |
| `03_tts_text_to_speech.md` | 9.5 KB | F5-TTS, Bark, XTTS, **SSML, Voice Cloning Ethics** |
| `04_speaker_diarization.md` | 7.6 KB | Pyannote, speaker embedding, **Overlap Speech Handling** |
| `05_voice_agents.md` | 9.9 KB | Latency budget, barge-in, WebSocket, **Streaming TTS, Turn-taking State Machine** |
| `06_production_speech.md` | 8.8 KB | Production deployment, scaling, edge cases |
| `speech_qa.md` | 5.7 KB | Q&A ôn tập |

**Điểm mạnh**: Augmentation full pipeline (SpecAugment + noise + pitch + speed). Streaming ASR class production-ready. Voice agents có turn-taking state machine (Mermaid).

**Kiến thức nổi bật**:
- SpecAugment: Frequency + Time masking (most important ASR augmentation)
- WER benchmarks: Whisper large-v3 ~3% (English), beats human ~4%
- Voice agent latency: Target <500ms end-to-end

---

## 07_advanced_rag — RAG Nâng Cao

| File | Size | Chủ đề chính |
|------|:----:|-------------|
| `01_chunking_strategies.md` | 11.4 KB | Recursive, semantic, parent-child chunking |
| `02_hybrid_search.md` | 9.7 KB | Dense + sparse retrieval, BM25 + embedding fusion |
| `03_reranking.md` | 10 KB | Cross-encoder reranking, Cohere Rerank, ColBERT |
| `04_multimodal_rag.md` | 12.9 KB | Image/audio RAG, **Text-to-SQL RAG** |
| `05_graph_rag.md` | 9.5 KB | Knowledge graphs, Neo4j, entity extraction |
| `06_rag_evaluation.md` | 14.4 KB | RAGAS framework, faithfulness, relevancy metrics |
| `rag_qa.md` | 5.3 KB | Q&A ôn tập |

**Điểm mạnh**: Text-to-SQL section mới với SQL sanitization + safety patterns. RAG evaluation comprehensive (RAGAS + custom metrics). Chunking strategies đa dạng.

**Kiến thức nổi bật**:
- Text-to-SQL: Schema injection + safe_execute (read-only, forbid DROP/DELETE)
- Hybrid search: BM25 (keyword) + Dense (semantic) → Reciprocal Rank Fusion
- RAGAS: faithfulness, answer_relevancy, context_precision, context_recall

---

## 08_fullstack_ai — Fullstack AI Application

| File | Size | Chủ đề chính |
|------|:----:|-------------|
| `01_api_design.md` | 11.5 KB | FastAPI for AI, async endpoints, middleware |
| `02_streaming_sse.md` | 13.8 KB | SSE vs WebSocket, OpenAI streaming, **Edge Cases (retry, cancel, partial JSON)** |
| `03_auth_security.md` | 11.8 KB | JWT, API keys, rate limiting, CORS |
| `04_frontend_ai.md` | 12.8 KB | React chat UI, markdown rendering, code highlighting |
| `05_state_management.md` | 10.1 KB | Zustand, conversation management, **Undo/Redo, Offline Queue** |
| `06_deployment.md` | 13.1 KB | Docker, Fly.io, Vercel, CI/CD for web apps |
| `fullstack_qa.md` | 5.3 KB | Q&A ôn tập |

**Điểm mạnh**: Streaming edge cases production-grade (AbortController, auto-retry with backoff). State management có undo/redo pattern cho regenerate response. Offline-first queue pattern.

**Kiến thức nổi bật**:
- SSE streaming: Server-Sent Events cho AI chat (simpler than WebSocket)
- AbortController: Cancel streaming mid-response ("Stop generating")
- Offline Queue: Queue messages when offline, flush when back online

---

## 09_interview_prep — Chuẩn Bị Phỏng Vấn

| File | Size | Chủ đề chính |
|------|:----:|-------------|
| `01_technical_questions.md` | 31.4 KB | **110 câu hỏi kỹ thuật** (100 core + **10 trends 2026**) |
| `02_behavioral_questions.md` | 7.6 KB | STAR method, common behavioral questions |
| `03_coding_challenges.md` | 23.3 KB | Python coding, ML coding, SQL challenges |
| `04_system_design_cases.md` | 10.1 KB | 5 real cases + **Multi-Tenant AI Platform** |
| `05_portfolio_presentation.md` | 7.7 KB | Portfolio structure, presentation tips |

**Điểm mạnh**: 110 câu hỏi bao phủ toàn bộ AI stack. 2026 trends cập nhật (MCP, reasoning models, structured output). System design có 6 cases thực tế.

**Kiến thức nổi bật**:
- 10 trending topics 2026: Reasoning models, MCP, long context vs RAG, multimodal
- Multi-tenant AI: Data isolation, cost allocation, prompt leakage prevention
- System design framework: Requirements → Architecture → Trade-offs → Monitoring

---

## 📊 Thống Kê Tổng Quan

| Module | Files | Size | Mức Độ |
|--------|:-----:|:----:|:------:|
| 00_foundations | 10 | 164 KB | ⭐⭐⭐⭐⭐ |
| 01_classical_ml | 8 | 87.7 KB | ⭐⭐⭐⭐⭐ |
| 02_deep_learning_cv | 8 | 95.4 KB | ⭐⭐⭐⭐ |
| 03_nlp_llm | 9 | 121.5 KB | ⭐⭐⭐⭐⭐ |
| 04_mlops | 8 | 78.6 KB | ⭐⭐⭐⭐ |
| 05_ai_agents | 8 | 80.8 KB | ⭐⭐⭐⭐ |
| 06_speech_ai | 8 | 63.2 KB | ⭐⭐⭐⭐ |
| 07_advanced_rag | 8 | 74.2 KB | ⭐⭐⭐⭐ |
| 08_fullstack_ai | 8 | 79.6 KB | ⭐⭐⭐⭐ |
| 09_interview_prep | 6 | 81.3 KB | ⭐⭐⭐⭐⭐ |
| **TOTAL** | **83** | **~927 KB** | **Expert** |
