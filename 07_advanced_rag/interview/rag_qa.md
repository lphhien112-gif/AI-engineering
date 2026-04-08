# 🎯 Advanced RAG — Câu Hỏi Phỏng Vấn (25+)

---

## Chunking (5 câu)

### Q1: Optimal chunk size?
**A**: Phụ thuộc use case. Q&A: 500-1000 tokens (precise). Summarization: 1500-3000 (broader). Code: function/class level. Always benchmark with your data.

### Q2: Recursive vs semantic chunking?
**A**: Recursive: split by separators (¶ → sentence → word), deterministic, fast. Semantic: embed sentences, split where similarity drops, better quality nhưng cần embedding model và chậm hơn.

### Q3: Overlap tại sao?
**A**: Prevent info loss at chunk boundaries. 10-20% of chunk size. Too much → duplicate retrievals. Too little → context breaks across chunks.

### Q4: Parent-child chunking?
**A**: Small child chunks cho embedding/retrieval (precise matching). Return larger parent chunk cho LLM (full context). Best of both worlds: precise retrieval + rich context.

### Q5: Agentic chunking?
**A**: LLM quyết định chunk boundaries. Process: read proposition-by-proposition → LLM decides "same topic or new topic?" → group into chunks. Highest quality, most expensive.

---

## Hybrid Search (5 câu)

### Q6: Hybrid search tại sao?
**A**: Vector: semantic similarity (good for meaning). BM25: keyword/exact match (good for names, codes, IDs). Hybrid combines → robust across query types.

### Q7: RRF vs weighted fusion?
**A**: RRF: rank-based (1/(k+rank)), no normalization needed. Weighted: score-based, need normalization (min-max). RRF simpler, more robust to score distribution differences.

### Q8: Alpha parameter tuning?
**A**: alpha=0.7 (more vector) cho semantic queries. alpha=0.3 (more BM25) cho exact match. Tune on labeled query set. Start at 0.5.

### Q9: HyDE (Hypothetical Document Embeddings)?
**A**: Generate hypothetical answer to query → embed that answer → search with that embedding. Why? Answer is more similar to documents than question is. Improves recall.

### Q10: Query expansion?
**A**: Generate multiple reformulated queries → retrieve for each → merge results. Captures different aspects of user intent. Use LLM to generate 3-5 variants.

---

## Reranking (4 câu)

### Q11: Two-stage retrieval?
**A**: Stage 1: Fast retrieval (bi-encoder, top 50-100). Stage 2: Accurate reranking (cross-encoder, top 5-10). Better precision without sacrificing recall.

### Q12: Bi-encoder vs cross-encoder?
**A**: Bi: encode query and doc separately → cosine similarity. Fast O(1). Cross: encode [query; doc] together → direct score. Accurate O(N). Use bi for retrieval, cross for reranking.

### Q13: ColBERT?
**A**: Late interaction model. Token-level embeddings stored per doc. At query time: compute MaxSim per token pair. Faster than cross-encoder, better than bi-encoder. Middle ground.

### Q14: Reranking latency budget?
**A**: Reranking 50 candidates ≈ 100-300ms (cross-encoder on GPU). Acceptable for most applications. Reranking 100+ → batch + GPU essential.

---

## Multi-modal RAG (4 câu)

### Q15: Multi-modal RAG challenges?
**A**: (1) Parsing complex layouts (PDF tables, figures), (2) Cross-modal embedding alignment, (3) Multi-modal LLM context, (4) Different chunking per modality.

### Q16: Table handling trong RAG?
**A**: (1) Convert to natural language for embedding, (2) Store structured data for SQL queries, (3) Summarize table for retrieval, (4) Return original table to LLM. Store multiple representations.

### Q17: Image RAG?
**A**: Vision model generates text description → embed description → retrieve. When generating answer: pass original image + context to multi-modal LLM. Caption quality critical.

### Q18: PDF parsing best practices?
**A**: Unstructured: best quality, layout-aware. PyMuPDF: fastest. LlamaParse: cloud API. Always: separate text, tables, images. Preserve metadata (page, section).

---

## Graph RAG (3 câu)

### Q19: Graph RAG khi nào cần?
**A**: Multi-hop reasoning ("Who works at the company that invented X?"), aggregation ("main themes?"), entity-centric queries. Complements vector RAG, not replaces.

### Q20: Knowledge graph construction?
**A**: LLM extracts (entity, relationship, entity) triples from chunks. Challenges: entity resolution ("OpenAI" = "Open AI"), consistency, hallucinated relationships.

### Q21: Microsoft GraphRAG approach?
**A**: Build KG → community detection (Leiden algorithm) → generate community summaries → at query time: retrieve relevant communities for global questions. SOTA for summarization queries.

---

## Evaluation (4 câu)

### Q22: RAGAS metrics?
**A**: Faithfulness (grounded in context?), Answer Relevancy (addresses question?), Context Precision (relevant chunks?), Context Recall (all info retrieved?). LLM-as-judge evaluates each.

### Q23: Faithfulness tại sao quan trọng?
**A**: Low faithfulness = hallucination (answer not in context). Critical for trust. Evaluate: decompose answer into claims → check each claim against context. Target: >0.9.

### Q24: Regression testing cho RAG?
**A**: Golden test set (100+ question/answer pairs). Run after every change (new embedding model, chunking, prompt). Alert if any metric drops >2%. Prevent silent degradation.

### Q25: LLM-as-Judge vs human eval?
**A**: LLM: fast, scalable, consistent. Human: nuanced, catches edge cases, costly. Hybrid: LLM for regression testing, human for periodic validation. Calibrate LLM-judge against human scores.
