# 🤖 AI Review Prompts — Knowledge Base Self-Review

> **Hướng dẫn sử dụng**: Copy prompt cho từng folder, paste vào AI (ChatGPT/Claude/Gemini), kèm toàn bộ nội dung các file .md trong folder đó để AI review chất lượng kiến thức.
>
> **Quy trình**: 
> 1. Chọn folder cần review (vd: `00_foundations`)
> 2. Copy prompt tương ứng bên dưới
> 3. Attach/paste toàn bộ file `.md` trong folder đó
> 4. AI sẽ review theo các tiêu chí đã định sẵn
> 5. Thực hiện cải thiện dựa trên feedback

---

## Folder 00: Foundations

```
Bạn là AI Reviewer chuyên gia về Software Engineering Foundations. 

Tôi sẽ cung cấp toàn bộ nội dung kiến thức trong folder "00_foundations" — bao gồm các chủ đề: Python basics, Python advanced, Math for AI, Git, Docker, API Design, SQL, Data Engineering.

Hãy review theo các tiêu chí sau:

### 1. Độ chính xác kỹ thuật (Technical Accuracy)
- Code snippets có chạy đúng không? Có lỗi syntax, logic, hoặc deprecated API không?
- Công thức toán (đặc biệt phần Attention Math, Optimizer) có chính xác không?
- Các giải thích có đúng về mặt khoa học/kỹ thuật không?

### 2. Độ sâu kiến thức (Depth)
- Mỗi chủ đề đã đạt mức "AI Engineer có 1-2 năm kinh nghiệm" chưa?
- Có thiếu khái niệm quan trọng nào không? (ví dụ: async/await có đủ chi tiết? Docker networking?)
- Phần Math: Attention math numerical example có rõ ràng step-by-step không?
- Phần Optimizer: Adam vs AdamW vs SGD comparison có đủ chi tiết production-level không?

### 3. Tính ứng dụng (Practical Relevance)
- Code có phải production-grade không? (error handling, typing, docstrings)
- Có đủ "khi nào dùng cái gì" (decision guide) không?
- Các ví dụ có sát thực tế AI engineer hàng ngày không?

### 4. Interview Readiness
- Phần Q&A (foundations_qa.md) có cover đủ câu hỏi hay gặp không?
- Câu trả lời có đủ sâu để impress interviewer không?
- Có thiếu follow-up questions quan trọng không?

### 5. Cấu trúc & Format
- Có nhất quán format giữa các files không? 
- Mermaid diagrams có render đúng không?
- Có section nào quá dài hoặc quá ngắn không?

### Output Format
Trả lời theo format:

**📊 Điểm tổng: X/10**

**✅ Điểm mạnh:**
- ...

**⚠️ Cần cải thiện:**
| Vấn đề | File | Mức độ | Gợi ý sửa |
|--------|------|--------|-----------|
| ... | ... | Critical/Medium/Low | ... |

**🔧 Top 5 cải thiện ưu tiên:**
1. ...
2. ...
```

---

## Folder 01: Classical ML

```
Bạn là AI Reviewer chuyên gia về Machine Learning cổ điển (Classical ML).

Tôi sẽ cung cấp toàn bộ nội dung kiến thức trong folder "01_classical_ml" — bao gồm: Supervised Learning, Unsupervised Learning, Feature Engineering, Evaluation Metrics, Hyperparameter Tuning, Practical Pipeline.

Hãy review theo các tiêu chí sau:

### 1. Độ chính xác kỹ thuật
- Sklearn API usage có đúng phiên bản mới nhất không?
- Gradient Boosting residual learning explanation có chính xác toán học không?
- XGBoost vs LightGBM vs CatBoost comparison có chính xác không?
- Probability Calibration (Brier score, Platt scaling, Isotonic) có đúng không?

### 2. Độ sâu kiến thức
- Gradient Boosting: Có giải thích residual learning step-by-step (F₀, h₁, h₂, ...) không?
- Ensemble methods: Bagging vs Boosting có đủ rõ trade-off không?
- Evaluation: Có thiếu metric quan trọng nào không? (calibration, lift chart, gain chart?)
- Optuna tuning: objective function có best practice không?

### 3. Tính thực tế
- Pipeline code có production-ready không? (sklearn Pipeline, ColumnTransformer)
- Feature engineering có cover đủ case thực tế không? (time series features, text features)
- Có case study hoặc benchmark comparison không?

### 4. Interview Readiness
- "Tại sao Random Forest resist overfitting?" — có trả lời được không?
- "Giải thích Gradient Boosting bằng toán?" — có numerical example không?
- "Khi nào dùng XGBoost vs LightGBM vs CatBoost?" — có decision guide không?

### 5. Gaps Analysis
- So với Kaggle top solutions, có thiếu trick nào không? (target encoding, stacking, blending)
- So với production ML systems, có thiếu gì không? (feature stores, A/B testing, monitoring)

### Output Format
**📊 Điểm tổng: X/10**

**✅ Điểm mạnh:**
- ...

**⚠️ Cần cải thiện:**
| Vấn đề | File | Mức độ | Gợi ý sửa |
|--------|------|--------|-----------|
| ... | ... | Critical/Medium/Low | ... |

**🔧 Top 5 cải thiện ưu tiên:**
1. ...
```

---

## Folder 02: Deep Learning & Computer Vision

```
Bạn là AI Reviewer chuyên gia về Deep Learning và Computer Vision.

Tôi sẽ cung cấp toàn bộ nội dung folder "02_deep_learning_cv" — bao gồm: Neural Networks, CNN Architectures, Transfer Learning, Segmentation & Detection, Training Recipes, Model Deployment.

Hãy review theo các tiêu chí sau:

### 1. Độ chính xác kỹ thuật
- Backpropagation explanation có đúng chain rule không?
- CNN architecture evolution (LeNet → AlexNet → VGG → ResNet → EfficientNet) có chính xác năm, params không?
- YOLO vs Mask R-CNN comparison có cập nhật không? (YOLOv8, YOLOv11?)
- Quantization: INT8, FP16, bfloat16 explanation có đúng không?

### 2. Độ sâu kiến thức
- ResNet skip connections: Có giải thích TẠI SAO giải quyết vanishing gradient (toán học) không?
- U-Net: Có đủ chi tiết encoder-decoder + skip connections cho segmentation không?
- Transfer learning: Có cover "which layers to freeze" strategies không?
- Training recipes: Mixed precision, gradient accumulation, learning rate warmup có đủ không?

### 3. Thực tế & Code
- PyTorch code có idiomatic không? (nn.Module, DataLoader, training loop)
- Model deployment: ONNX export + TensorRT có working code không?
- Data augmentation: có Albumentations examples không?
- Có benchmark numbers (accuracy, inference speed) cho các architectures không?

### 4. Thiếu sót
- Có thiếu Vision Transformer (ViT) không?
- Foundation models (SAM, DINOv2) có được đề cập không?
- 3D vision, video understanding có cần không?

### Output Format
**📊 Điểm tổng: X/10**

**✅ Điểm mạnh:**
- ...

**⚠️ Cần cải thiện:**
| Vấn đề | File | Mức độ | Gợi ý sửa |
|--------|------|--------|-----------|
| ... | ... | Critical/Medium/Low | ... |

**🔧 Top 5 cải thiện ưu tiên:**
1. ...
```

---

## Folder 03: NLP & LLM

```
Bạn là AI Reviewer chuyên gia về NLP và Large Language Models.

Tôi sẽ cung cấp toàn bộ nội dung folder "03_nlp_llm" — bao gồm: Transformers, LLM Fundamentals, Prompt Engineering, RAG Architecture, Fine-tuning, Evaluation, Vector Databases.

Hãy review theo các tiêu chí sau:

### 1. Độ chính xác kỹ thuật
- Self-attention formula có đúng không? Q×K^T/√d_k → softmax → ×V
- Flash Attention: Tiling algorithm, HBM vs SRAM explanation có chính xác không?
- LoRA/QLoRA: rank, alpha, trainable params calculation có đúng không?
- Structured Output: JSON Schema, Instructor, Function Calling usage có đúng API không?
- Tokenization: BPE, WordPiece, SentencePiece differences có chính xác không?

### 2. Độ sâu kiến thức
- Flash Attention: v1 vs v2 comparison đủ chi tiết không?
- Prompt Engineering: Chain-of-Thought có phân biệt zero-shot CoT vs few-shot CoT không?
- RAG: Có cover naive RAG vs advanced RAG differences không?
- Fine-tuning evaluation: A/B comparison, LLM-as-Judge có blind evaluation không?
- Vector DB: ANN algorithms (HNSW, IVF, PQ) có so sánh trade-offs không?

### 3. Cập nhật 2025-2026
- Có đề cập GPT-4o, Claude 3.5, Gemini 2.0 không?
- Reasoning models (o1, o3, DeepSeek R1) có được bao phủ không?
- Structured output có dùng mới nhất API (response_format với JSON Schema) không?
- Fine-tuning: Có mention Unsloth, axolotl accelerations không?

### 4. Production Patterns
- Token counting + cost estimation có code không?
- Retry logic, rate limiting cho API calls có không?
- Streaming + function calling combined patterns?
- Embedding model selection guide (text-embedding-3-small vs large)?

### 5. Interview Readiness
- "Giải thích self-attention step by step" — có numerical example không?
- "Flash Attention tại sao nhanh hơn?" — có HBM/SRAM explanation không?
- "LoRA vs full fine-tuning trade-offs?" — có cost comparison không?

### Output Format
**📊 Điểm tổng: X/10**

**✅ Điểm mạnh:**
- ...

**⚠️ Cần cải thiện:**
| Vấn đề | File | Mức độ | Gợi ý sửa |
|--------|------|--------|-----------|
| ... | ... | Critical/Medium/Low | ... |

**🔧 Top 5 cải thiện ưu tiên:**
1. ...
```

---

## Folder 04: MLOps

```
Bạn là AI Reviewer chuyên gia về MLOps và ML Production Systems.

Tôi sẽ cung cấp toàn bộ nội dung folder "04_mlops" — bao gồm: Experiment Tracking, Data Versioning, Containerization, CI/CD for ML, Model Monitoring, Cloud Platforms.

Hãy review theo các tiêu chí sau:

### 1. Độ chính xác kỹ thuật
- MLflow API usage có đúng không? (mlflow.log_param, log_metric, log_artifact)
- W&B integration code có production-ready không?
- Docker for ML: GPU setup (nvidia-docker, CUDA base images) có đúng không?
- GitHub Actions workflow cho ML training có hoạt động không?
- LLM tracking: cost calculation, prompt versioning có chính xác không?

### 2. Độ sâu kiến thức
- Experiment tracking: MLflow vs W&B trade-offs đủ sâu không?
- Data versioning: DVC có cover remote storage (S3, GCS) không?
- Model monitoring: Data drift detection có dùng statistical tests đúng không?
- LLM tracking: LangSmith/LangFuse integration có đủ chi tiết không?
- A/B testing prompts: methodology có sound không?

### 3. Thực tế Production
- Có CI/CD pipeline hoàn chỉnh (lint → test → train → deploy) không?
- Model registry workflow (staging → production → archived) có không?
- Monitoring alerting (PagerDuty, Slack) có integration code không?
- Cost optimization strategies cho cloud ML infrastructure?

### 4. Thiếu sót
- Feature stores (Feast, Tecton) có được đề cập không?
- ML Platform comparison (SageMaker vs Vertex AI vs Azure ML) đủ chi tiết không?
- Có GitOps for ML patterns không?
- LLMOps specific tools (LangSmith, Helicone, Promptfoo) có không?

### Output Format
**📊 Điểm tổng: X/10**

**✅ Điểm mạnh:**
- ...

**⚠️ Cần cải thiện:**
| Vấn đề | File | Mức độ | Gợi ý sửa |
|--------|------|--------|-----------|
| ... | ... | Critical/Medium/Low | ... |

**🔧 Top 5 cải thiện ưu tiên:**
1. ...
```

---

## Folder 05: AI Agents

```
Bạn là AI Reviewer chuyên gia về AI Agents và Agentic Systems.

Tôi sẽ cung cấp toàn bộ nội dung folder "05_ai_agents" — bao gồm: Agent Fundamentals, LangGraph, Tool Design, Memory Systems, Multi-Agent, Agent Safety.

Hãy review theo các tiêu chí sau:

### 1. Độ chính xác kỹ thuật
- ReAct pattern implementation có đúng không?
- LangGraph: StateGraph, nodes, edges, conditional routing code có đúng API không?
- Tool schema (JSON Schema for function calling) có đúng format không?
- Agent evaluation framework: metrics calculation có đúng không?
- Multi-agent protocols có implementation đúng không?

### 2. Độ sâu kiến thức
- Agent vs RAG distinction có rõ ràng không?
- LangGraph vs CrewAI vs AutoGen comparison đủ dimensions không?
- Memory: conversation buffer vs summary vs vector memory trade-offs?
- Safety: prompt injection attack vectors và defense strategies đủ không?
- Agent evaluation: task_completion_rate, tool_accuracy metrics có standard không?

### 3. Cập nhật 2025-2026
- MCP (Model Context Protocol) có được đề cập không?
- OpenAI Agents SDK (formerly Swarm) có cập nhật không?
- Function calling vs tool_use across providers (OpenAI vs Anthropic vs Google)?
- Agent observability tools (LangSmith traces, AgentOps)?

### 4. Production Patterns
- Error recovery: What happens when tool fails mid-execution?
- Cost control: Có max_iterations, budget limits patterns không?
- Checkpointing: Save/restore agent state across sessions?
- Human-in-the-loop: Approval workflows for critical actions?

### Output Format
**📊 Điểm tổng: X/10**

**✅ Điểm mạnh:**
- ...

**⚠️ Cần cải thiện:**
| Vấn đề | File | Mức độ | Gợi ý sửa |
|--------|------|--------|-----------|
| ... | ... | Critical/Medium/Low | ... |

**🔧 Top 5 cải thiện ưu tiên:**
1. ...
```

---

## Folder 06: Speech AI

```
Bạn là AI Reviewer chuyên gia về Speech AI, bao gồm ASR, TTS, Speaker Diarization, và Voice Agents.

Tôi sẽ cung cấp toàn bộ nội dung folder "06_speech_ai" — bao gồm: Audio Fundamentals, ASR (Whisper), TTS, Speaker Diarization, Voice Agents, Production Speech.

Hãy review theo các tiêu chí sau:

### 1. Độ chính xác kỹ thuật
- Mel spectrogram computation có đúng (n_fft, hop_length, n_mels) không?
- SpecAugment parameters có reasonable không?
- Whisper architecture description có chính xác không?
- Faster-whisper CTranslate2 usage có đúng API không?
- Pyannote diarization pipeline code có hoạt động không?
- WER calculation formula (S+I+D / N) có đúng không?

### 2. Độ sâu kiến thức
- Audio fundamentals: Nyquist theorem, bit depth explanation đủ không?
- ASR: Streaming pipeline (VAD + buffering + transcription) có realistic không?
- TTS: F5-TTS vs Bark vs XTTS comparison đủ dimensions không?
- Diarization: overlap handling strategies có practical không?
- Voice agents: latency budget breakdown có realistic targets không?
- SSML: examples có cover production use cases không?

### 3. Cập nhật 2025-2026
- Whisper turbo / large-v3 turbo có được đề cập không?
- Multimodal audio (GPT-4o audio, Gemini audio) có không?
- Fish Speech, CosyVoice có trong TTS comparison không?
- End-to-end voice agents (ElevenLabs, Vapi, LiveKit) có không?

### 4. Production Patterns
- Audio preprocessing pipeline (resample, normalize, VAD) có hoàn chỉnh không?
- Batch transcription vs real-time streaming trade-offs?
- Voice agent WebSocket implementation có production-ready không?
- Edge deployment for ASR (on-device Whisper) có không?

### 5. Thiếu sót
- Voice cloning ethics section có balanced không?
- Speaker verification (1:1) vs identification (1:N) có phân biệt không?
- Emotion recognition / sentiment from speech có cần không?

### Output Format
**📊 Điểm tổng: X/10**

**✅ Điểm mạnh:**
- ...

**⚠️ Cần cải thiện:**
| Vấn đề | File | Mức độ | Gợi ý sửa |
|--------|------|--------|-----------|
| ... | ... | Critical/Medium/Low | ... |

**🔧 Top 5 cải thiện ưu tiên:**
1. ...
```

---

## Folder 07: Advanced RAG

```
Bạn là AI Reviewer chuyên gia về Retrieval-Augmented Generation (RAG) systems.

Tôi sẽ cung cấp toàn bộ nội dung folder "07_advanced_rag" — bao gồm: Chunking Strategies, Hybrid Search, Reranking, Multimodal RAG, Graph RAG, RAG Evaluation.

Hãy review theo các tiêu chí sau:

### 1. Độ chính xác kỹ thuật
- Chunking: recursive vs semantic vs parent-child implementation có đúng không?
- Hybrid search: BM25 + dense fusion (RRF formula) có chính xác không?
- Reranking: Cross-encoder vs bi-encoder distinction có đúng không?
- Text-to-SQL: SQL injection prevention có đủ robust không?
- Graph RAG: Neo4j Cypher queries có đúng syntax không?
- RAGAS metrics formulas có chính xác không?

### 2. Độ sâu kiến thức
- Chunking: chunk_size vs chunk_overlap trade-offs có numerical guidelines không?
- Hybrid search: Khi nào BM25 wins vs khi nào dense wins?
- Reranking: Cohere Rerank vs cross-encoder latency comparison?
- Text-to-SQL vs Vector RAG decision guide có rõ ràng không?
- Graph RAG: When is graph RAG better than vector RAG?
- Evaluation: RAGAS framework có cover đủ dimensions không?

### 3. Production Patterns
- Chunking pipeline có handle PDFs, HTML, markdown khác nhau không?
- Embedding model selection: text-embedding-3 vs Cohere vs open-source?
- Caching strategy cho RAG (query cache, embedding cache)?
- Multi-tenancy trong RAG (namespace isolation)?

### 4. Thiếu sót
- Agentic RAG (self-correcting, query decomposition) có không?
- Contextual retrieval (Anthropic's approach) có đề cập không?
- Late chunking / contextual embeddings có không?
- RAG pipeline observability (LangSmith traces)?

### Output Format
**📊 Điểm tổng: X/10**

**✅ Điểm mạnh:**
- ...

**⚠️ Cần cải thiện:**
| Vấn đề | File | Mức độ | Gợi ý sửa |
|--------|------|--------|-----------|
| ... | ... | Critical/Medium/Low | ... |

**🔧 Top 5 cải thiện ưu tiên:**
1. ...
```

---

## Folder 08: Fullstack AI

```
Bạn là AI Reviewer chuyên gia về Fullstack AI Application Development.

Tôi sẽ cung cấp toàn bộ nội dung folder "08_fullstack_ai" — bao gồm: API Design, Streaming SSE, Auth & Security, Frontend AI, State Management, Deployment.

Hãy review theo các tiêu chí sau:

### 1. Độ chính xác kỹ thuật
- FastAPI async endpoints có đúng pattern không?
- SSE (Server-Sent Events) implementation có đúng spec không?
- JWT auth flow có secure không? (token rotation, refresh tokens)
- React chat UI code có modern patterns không? (hooks, TypeScript)
- Zustand state management có idiomatic không?
- Streaming edge cases: AbortController, retry logic có đúng không?

### 2. Độ sâu kiến thức
- SSE vs WebSocket: trade-offs có đúng và đầy đủ không?
- Auth: API key vs JWT vs OAuth2 decision guide?
- Frontend: Markdown rendering + code highlighting có production không?
- State management: Undo/redo pattern có correct immutability không?
- Offline queue: sync logic có handle conflicts không?
- Deployment: Docker + reverse proxy + HTTPS setup?

### 3. Tính ứng dụng
- Có build được 1 ChatGPT-like app hoàn chỉnh từ files này không?
- Rate limiting implementation có production-grade không?
- CORS configuration có đúng cho SPA + API architecture không?
- Error handling: có unified error format + client-side handling không?

### 4. Thiếu sót
- WebSocket implementation (for real-time features beyond chat)?
- File upload handling (multipart, presigned URLs)?
- Database design for conversations (schema, indexing)?
- Performance: pagination, virtual scrolling cho long conversations?
- Testing: E2E tests cho streaming, auth flows?

### Output Format
**📊 Điểm tổng: X/10**

**✅ Điểm mạnh:**
- ...

**⚠️ Cần cải thiện:**
| Vấn đề | File | Mức độ | Gợi ý sửa |
|--------|------|--------|-----------|
| ... | ... | Critical/Medium/Low | ... |

**🔧 Top 5 cải thiện ưu tiên:**
1. ...
```

---

## Folder 09: Interview Prep

```
Bạn là AI Reviewer chuyên gia về AI Engineer Interview Preparation.

Tôi sẽ cung cấp toàn bộ nội dung folder "09_interview_prep" — bao gồm: 110 Technical Questions, Behavioral Questions, Coding Challenges, System Design Cases, Portfolio Presentation.

Hãy review theo các tiêu chí sau:

### 1. Độ chính xác
- 110 câu hỏi kỹ thuật: Mỗi câu trả lời có chính xác không?
- 2026 AI Trends (câu 101-110): Thông tin có cập nhật, không outdated không?
- Coding challenges: Solutions có optimal time/space complexity không?
- System design: Architecture diagrams có realistic không?

### 2. Độ bao phủ (Coverage)
Kiểm tra coverage theo các topic:

| Topic | Số câu | Đủ chưa? |
|-------|:------:|----------|
| ML Fundamentals | 20 | ? |
| Deep Learning | 15 | ? |
| NLP / LLM | 20 | ? |
| RAG & Vector DB | 15 | ? |
| System Design | 10 | ? |
| MLOps | 10 | ? |
| Coding / Python | 10 | ? |
| 2026 Trends | 10 | ? |

- Có thiếu topic quan trọng nào không? (Speech AI, Agents, Fullstack)
- Behavioral questions có cover đủ STAR scenarios không?

### 3. Chất lượng câu trả lời
- Mỗi answer có đủ 3 elements: Core concept + Example + Follow-up?
- System design answers có follow framework: Requirements → Architecture → Trade-offs?
- Coding solutions có edge case handling không?
- Có câu nào answer quá ngắn (< 2 dòng) cần expand không?

### 4. Interview Strategy
- Có advice cho từng round không? (phone screen vs onsite vs system design)
- Có "red flags to avoid" section không?
- Portfolio presentation: có template/structure rõ ràng không?
- Salary negotiation tips?

### 5. Thiếu sót
- Có mock interview scripts không?
- Có "30-60-90 day plan" cho new AI engineer role không?
- Company-specific prep (Google, Meta, startup) có không?
- Có practice schedule / study plan không?

### Output Format
**📊 Điểm tổng: X/10**

**✅ Điểm mạnh:**
- ...

**⚠️ Cần cải thiện:**
| Vấn đề | File | Mức độ | Gợi ý sửa |
|--------|------|--------|-----------|
| ... | ... | Critical/Medium/Low | ... |

**🔧 Top 5 cải thiện ưu tiên:**
1. ...
```

---

## 📝 Hướng Dẫn Sử Dụng Nâng Cao

### Cách attach files cho AI review

**Option 1: Copy-paste trực tiếp**
```
1. Mở terminal tại folder cần review
2. Chạy: cat docs/*.md (Mac/Linux) hoặc Get-Content docs\*.md (Windows)
3. Copy output → paste vào chat AI cùng prompt
```

**Option 2: Upload files**
```
1. Zip folder docs/ 
2. Upload lên Claude/ChatGPT (nếu hỗ trợ file upload)
3. Paste prompt tương ứng
```

**Option 3: Dùng AI coding assistant (Cursor/Windsurf)**
```
1. Mở folder trong IDE
2. Select all .md files trong folder
3. Paste prompt vào chat → AI tự đọc files
```

### Review theo batch

Nếu folder quá lớn (>100KB), chia ra review theo nhóm:

```
Batch 1: Files 01-03 (core concepts)
Batch 2: Files 04-06 (advanced topics)
Batch 3: QA + README (review consistency)
```

### Tracking improvements

Sau mỗi lần review, tạo issue list:
```
| # | File | Issue | Priority | Status |
|---|------|-------|----------|--------|
| 1 | 02_math_for_ai.md | Missing SVD application example | Medium | ⬜ |
| 2 | 01_supervised_learning.md | Stacking ensemble not covered | Low | ⬜ |
```
