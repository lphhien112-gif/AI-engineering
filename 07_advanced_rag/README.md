# 🔍 07 — Advanced RAG

> Beyond basic RAG: Chunking strategies, Hybrid Search, Reranking, Multi-modal, Graph RAG, Evaluation.

## 📚 Docs

| File | Chủ đề | Thời gian đọc |
|------|--------|---------------|
| [01_chunking_strategies.md](docs/01_chunking_strategies.md) | Semantic, Recursive, Agentic Chunking | ~12 min |
| [02_hybrid_search.md](docs/02_hybrid_search.md) | BM25 + Vector, RRF Fusion, Query Expansion | ~12 min |
| [03_reranking.md](docs/03_reranking.md) | Cross-encoder, Cohere Rerank, ColBERT | ~10 min |
| [04_multimodal_rag.md](docs/04_multimodal_rag.md) | Image, Table, PDF parsing, Vision RAG | ~12 min |
| [05_graph_rag.md](docs/05_graph_rag.md) | Knowledge Graphs, Neo4j, Entity Resolution | ~12 min |
| [06_rag_evaluation.md](docs/06_rag_evaluation.md) | RAGAS, LLM-as-Judge, Regression Testing | ~10 min |

## 💻 Examples

```bash
cd 07_advanced_rag/examples
python chunking_demo.py        # Chunking strategies comparison
python hybrid_search.py        # BM25 + Vector hybrid search
python rag_evaluation.py       # RAGAS-style evaluation
```

## ✅ Checklist
- [ ] Đọc hết 6 docs
- [ ] Chạy 3 examples
- [ ] Trả lời 25+ câu trong `interview/rag_qa.md`
