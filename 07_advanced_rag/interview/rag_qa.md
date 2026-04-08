# 🎯 Advanced RAG — Câu Hỏi Phỏng Vấn (35+)

> Mỗi câu quan trọng có: giải thích → decision guide / ví dụ cụ thể → code → follow-up.

---

## Chunking (6 câu)

### Q1: Optimal chunk size — chọn thế nào?
**A**: Không có magic number. Phụ thuộc **retrieval task** + **document type**.

| Use Case | Chunk Size | Tại sao |
|----------|:----------:|---------|
| Factual Q&A | 200-500 tokens | Precise matching, ít noise |
| Summarization | 1000-3000 tokens | Cần broader context |
| Legal/Medical | 500-1000 tokens | Paragraph-level accuracy |
| Code docs | Function/class level | Structural boundaries |
| Chat/FAQ | 100-300 tokens | Short, direct answers |

**Benchmark approach**: test 3-4 sizes on your data, measure retrieval recall@10. Typical finding: 500 tokens = sweet spot for general Q&A.

- **Follow-up**: "Chunk overlap bao nhiêu?" → 10-20% of chunk size. Too much → duplicate retrievals inflate results. Too little → context breaks at boundaries.

### Q2: Recursive vs semantic chunking?
**A**: 
```
Recursive:  Split by separators: ¶ → \n\n → \n → sentence → word
            ✅ Fast, deterministic, no model needed
            ❌ May split mid-concept

Semantic:   Embed each sentence → compute similarity → split where similarity drops
            ✅ Better topic coherence within chunks
            ❌ Slower (need embedding model), non-deterministic
```

**Decision guide**: 
- Structured docs (markdown, HTML) → Recursive (respect structure)
- Unstructured prose (essays, reports) → Semantic (respect meaning)
- Production default → Recursive (reliable, debuggable)

### Q3: Parent-child chunking?
**A**: Best of both worlds — precise retrieval + rich context.
```
Document
├── Parent chunk (1000 tokens) — returned to LLM for context
│   ├── Child chunk (200 tokens) — used for embedding/retrieval
│   ├── Child chunk (200 tokens) — used for embedding/retrieval
│   └── Child chunk (200 tokens) — used for embedding/retrieval
```
- Retrieve on **child** (small = precise matching)
- Return **parent** to LLM (large = full context)
- **Implementation**: LlamaIndex `SentenceWindowNodeParser`, LangChain `ParentDocumentRetriever`

### Q4: Contextual chunking (Anthropic)?
**A**: Prepend context BEFORE embedding each chunk.

```python
# Standard: embed chunk alone
chunk = "The transformer uses multi-head attention..."
embedding = embed(chunk)  # Chunk has no surrounding context

# Contextual: LLM generates context → prepend to chunk
context = llm("Summarize what this chunk is about in the document: ...")
# "This chunk is from Chapter 3 of a paper about neural architectures,
#  discussing the attention mechanism."
enriched = f"{context}\n\n{chunk}"
embedding = embed(enriched)  # Much better semantic representation
```
- **Result**: ~49% reduction in retrieval failures (Anthropic's benchmark)
- **Cost**: extra LLM call per chunk (use cheap model: GPT-4o-mini)

### Q5: Agentic chunking?
**A**: LLM decides chunk boundaries. Process proposition-by-proposition → "same topic or new topic?"
- Highest quality boundaries, most expensive
- **When**: critical knowledge base (legal, medical) where wrong chunks = wrong answers
- **Practical alternative**: semantic chunking (80% quality, 10% cost)

### Q6: Late chunking (Jina)?
**A**: Embed **full document first** → split into chunks AFTER embedding.
- Traditional: chunk → embed each independently (lose document context)
- Late: embed full doc (captures cross-reference) → split embeddings into chunk-level
- **Benefit**: each chunk embedding aware of full document context
- **Limitation**: document length limited by embedding model context window

---

## Hybrid Search (6 câu)

### Q7: Tại sao hybrid search?
**A**: Neither vector nor keyword search is perfect alone.

```
Query: "Error code E1234 in production"
  BM25: ✅ Exact match "E1234" (keyword wins here)
  Vector: ❌ "E1234" not semantically meaningful

Query: "how to fix authentication issues"  
  BM25: ❌ No exact keywords in docs (docs say "login failures")
  Vector: ✅ Semantic similarity catches "auth ≈ login" 

Hybrid: ✅ Both queries handled well
```

### Q8: RRF vs weighted fusion?
**A**: 

| | RRF (Reciprocal Rank Fusion) | Weighted Fusion |
|-|------------------------------|----------------|
| **Formula** | score = Σ 1/(k + rank_i) | score = α × vector + (1-α) × bm25 |
| **Normalization** | Not needed (rank-based) | Required (different score scales) |
| **Robustness** | Better (rank stable) | Sensitive to score distribution |
| **Tuning** | k parameter (usually 60) | α parameter (0-1) |

- **Default**: RRF with k=60 (simpler, more robust)
- **When weighted**: when you have labeled data to tune α precisely

### Q9: Alpha parameter tuning?
**A**: α controls vector vs BM25 weight in hybrid search.
- α = 0.7 → more semantic (good for natural language questions)
- α = 0.3 → more keyword (good for codes, IDs, technical terms)
- **How to tune**: labeled query set → measure Recall@10 for different α → pick best
- **Start**: α = 0.5 (equal weight) → adjust based on query analysis
- **In practice**: Weaviate, Qdrant support hybrid search with α parameter natively.

### Q10: HyDE (Hypothetical Document Embeddings)?
**A**: Generate hypothetical answer → embed that → search.

```python
def hyde_search(query: str, k: int = 5):
    # 1. LLM generates hypothetical answer
    hypothetical = llm.generate(f"Write a paragraph answering: {query}")
    
    # 2. Embed hypothetical (more similar to docs than question is)
    query_embedding = embed(hypothetical)
    
    # 3. Search with hypothetical embedding
    return vector_db.search(query_embedding, top_k=k)
```
- **Why works**: "What is attention?" (question) vs "Attention is a mechanism that..." (document). HyDE answer is in document-space → better similarity.
- **Cost**: 1 extra LLM call per query. Worth it for complex questions.

### Q11: Query expansion?
**A**: Generate multiple reformulations → retrieve for each → merge results.
```python
# Single query → misses some relevant docs
results = search("machine learning deployment")

# Expanded → captures different aspects
queries = llm.generate("Generate 3 variants: 'ML deployment'")
# ["ML model serving in production", 
#  "deploying machine learning pipelines",
#  "MLOps deployment strategies"]
all_results = [search(q) for q in queries]
merged = reciprocal_rank_fusion(all_results)  # Merge + deduplicate
```
- **When**: complex queries with multiple aspects. Overkill for simple lookups.

### Q12: Multi-index strategies?
**A**: Different doc types → different indexes optimized for each.
- **Code docs**: chunk by function/class, code-specific embedding model
- **API docs**: structured chunks with endpoint + params
- **Text docs**: semantic chunking, general embedding
- **Tables**: convert to natural language + keep structured format
- **Router**: classify query → route to appropriate index

---

## Reranking (4 câu)

### Q13: Two-stage retrieval — tại sao?
**A**: 
```
Stage 1: Fast retrieval (bi-encoder)
  → Retrieve top 50-100 candidates
  → Speed: 1ms per 1M docs (precomputed embeddings)

Stage 2: Accurate reranking (cross-encoder)
  → Score top 50 → return top 5-10
  → Speed: 100-300ms for 50 candidates
```
- **Why not cross-encoder for everything?** O(N) — đọc mỗi doc cùng query. 1M docs = impossible.
- **Why not bi-encoder only?** Misses nuances. Cross-encoder sees query+doc TOGETHER → catches relationships.

### Q14: Bi-encoder vs cross-encoder?
**A**: 

| | Bi-encoder | Cross-encoder |
|-|-----------|--------------|
| **Encoding** | Query + doc separately → cosine | [query; doc] together → direct score |
| **Speed** | O(1) lookup (precomputed) | O(N) inference |
| **Accuracy** | Good | Better (sees interactions) |
| **Use case** | Retrieval (search) | Reranking (refine top-K) |
| **Models** | text-embedding-3, BGE, E5 | ms-marco-MiniLM, BGE-reranker |

### Q15: ColBERT?
**A**: Middle ground — token-level late interaction.
- Store per-token embeddings for each doc (more storage)
- At query time: compute MaxSim per token pair between query and doc
- **Faster** than cross-encoder (precompute doc tokens), **better** than bi-encoder
- Trade-off: 10-50× more storage than bi-encoder

### Q16: Reranking latency budget?
**A**: 
- 50 candidates × cross-encoder: 100-300ms (GPU). Acceptable.
- 100+ candidates: batch + GPU essential, or use ColBERT
- **Production pattern**: retrieve 50 → rerank → return top 5. Total: retrieval (10ms) + rerank (200ms) = 210ms.
- **Follow-up**: "Cohere Rerank vs open-source?" → Cohere: API-based, easy, ~100ms. Open-source: self-host, more control, similar quality (BGE-reranker).

---

## Multi-modal RAG (5 câu)

### Q17: Multi-modal RAG challenges?
**A**: 4 core challenges:
1. **Parsing**: complex layouts (PDF tables, figures, multi-column) → lossy conversion
2. **Cross-modal embedding**: image + text in same vector space? → imperfect alignment
3. **Context window**: images as tokens (expensive) vs text descriptions (lossy)
4. **Different chunking**: text = paragraphs, images = whole, tables = rows vs whole

### Q18: Table handling trong RAG?
**A**: Store **multiple representations** per table:
1. **Natural language summary**: "This table shows Q1-Q4 revenue by region" → for retrieval
2. **Structured data**: original CSV/JSON → for SQL queries
3. **Markdown format**: table as markdown → for LLM context
- Retrieve on summary (semantic). Return markdown + summary to LLM.

### Q19: Image RAG?
**A**: Two approaches:
- **Caption-based**: Vision model generates description → embed description → search by text
- **Multi-modal embedding**: CLIP/SigLIP embeds image directly → search in shared space
- **Generation**: retrieve original image → pass to multimodal LLM (GPT-4o, Gemini) + text context
- **Follow-up**: "CLIP limitations?" → trained on internet image-text pairs, weak on domain-specific (medical, satellite). Fine-tune or use domain-specific models.

### Q20: PDF parsing best practices?
**A**: 

| Tool | Speed | Layout Quality | Tables | Images |
|------|:-----:|:--------------:|:------:|:------:|
| **Unstructured** | Medium | Best | ✅ | ✅ |
| **PyMuPDF** | Fastest | Good | ⚠️ | ⚠️ |
| **LlamaParse** | Slow (API) | Excellent | ✅ | ✅ |
| **Docling** (IBM) | Medium | Very good | ✅ | ✅ |

- **Always**: separate text, tables, images. Preserve metadata (page number, section).
- **⚠️ Gotcha**: scanned PDFs need OCR first (Tesseract, EasyOCR).

### Q21: Text-to-SQL RAG?
**A**: Natural language → SQL query → execute → return results.
```python
def text_to_sql_rag(question: str, schema: str):
    # 1. Generate SQL from question + schema
    sql = llm.generate(f"""
        Given this schema:
        {schema}
        
        Generate SQL for: {question}
        Return ONLY the SQL query. Use SELECT only (no INSERT/UPDATE/DELETE).
    """)
    
    # 2. Safety: validate SQL (prevent injection)
    if any(word in sql.upper() for word in ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER"]):
        return "Unsafe query rejected"
    
    # 3. Execute (read-only connection!)
    results = db.execute(sql, read_only=True, timeout=5)
    
    # 4. Generate natural language answer
    return llm.generate(f"Given SQL results: {results}\nAnswer: {question}")
```
- **When**: structured data + analytical questions ("total revenue by quarter")
- **vs Vector RAG**: Text-to-SQL for aggregation/counting. Vector RAG for semantic search.

---

## Graph RAG (4 câu)

### Q22: Graph RAG khi nào cần?
**A**: 

| Query Type | Vector RAG | Graph RAG |
|-----------|:----------:|:---------:|
| "What is X?" | ✅ | ⚠️ overkill |
| "How does X relate to Y?" | ⚠️ weak | ✅ |
| "Who works at the company that made Z?" | ❌ multi-hop | ✅ |
| "What are the main themes?" | ❌ aggregation | ✅ |
| "Find documents about topic X" | ✅ | ⚠️ |

- **Rule**: multi-hop reasoning, relationship queries, aggregation → Graph RAG
- **Complement**: Graph RAG + Vector RAG together (best of both)

### Q23: Knowledge graph construction?
**A**: LLM extracts (entity, relationship, entity) triples from chunks.
```python
# Extract triples from text
triples = llm.extract("""
    "OpenAI released GPT-4o in May 2024. Sam Altman is the CEO."
    → (OpenAI, released, GPT-4o)
    → (GPT-4o, released_date, May 2024)
    → (Sam Altman, is_CEO_of, OpenAI)
""")
```
**Challenges**: entity resolution ("OpenAI" = "Open AI" = "openai"), consistency, hallucinated relationships.

### Q24: Microsoft GraphRAG approach?
**A**: 
1. Build knowledge graph from all documents
2. **Community detection** (Leiden algorithm) → group related entities
3. Generate **community summaries** (abstract each cluster)
4. At query time: retrieve relevant communities → use summaries for global reasoning

**Strength**: global questions ("What are the main themes across all docs?") — impossible for vector RAG.
**Weakness**: expensive to build (many LLM calls), needs rebuilding when data changes.

### Q25: Graph database options?
**A**: 
- **Neo4j**: most popular, Cypher query language, good community
- **Amazon Neptune**: managed, AWS-native
- **FalkorDB**: Redis-compatible, fast for production
- For RAG specifically: Neo4j + LangChain integration most mature. Alternative: lightweight in-memory graph (NetworkX) for small KGs.

---

## Evaluation (5 câu)

### Q26: RAGAS metrics chi tiết?
**A**: 4 dimensions, LLM-as-judge evaluates each:

| Metric | Question | Formula idea | Good score |
|--------|----------|-------------|:----------:|
| **Faithfulness** | Answer grounded in context? | claims_in_context / total_claims | > 0.9 |
| **Answer Relevancy** | Answers the question? | avg cosine(answer, question variants) | > 0.8 |
| **Context Precision** | Retrieved chunks relevant? | relevant_chunks_before_irrelevant / total | > 0.7 |
| **Context Recall** | All needed info retrieved? | claims_answered / total_claims_needed | > 0.8 |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset,  # HuggingFace Dataset with question, answer, contexts, ground_truth
    metrics=[faithfulness, answer_relevancy, context_precision],
)
print(result)  # {'faithfulness': 0.92, 'answer_relevancy': 0.85, ...}
```

### Q27: Faithfulness tại sao quan trọng nhất?
**A**: Low faithfulness = **hallucination** (answer claims something NOT in retrieved context).
- **Evaluate step-by-step**:
  1. Decompose answer into individual claims
  2. Check each claim against context: supported? unsupported? contradicted?
  3. Faithfulness = supported_claims / total_claims
- **Target**: > 0.95 for production (especially legal, medical, finance)
- **Fix low faithfulness**: better prompting ("Only use provided context"), improve retrieval (get the right context)

### Q28: Regression testing cho RAG?
**A**: Golden test set (100+ question/answer/context triples). Run after **every** change.

```python
# Changes that need regression testing:
# - New embedding model
# - Chunking strategy change
# - Prompt template update  
# - New data ingested
# - Reranker model swap

def regression_test(golden_set, pipeline):
    scores = evaluate(golden_set, pipeline)
    baseline = load_baseline_scores()
    
    for metric, score in scores.items():
        if score < baseline[metric] - 0.02:  # 2% tolerance
            raise RegressionError(f"{metric} dropped: {baseline[metric]:.3f} → {score:.3f}")
    
    save_as_new_baseline(scores)  # Update if all pass
```

### Q29: LLM-as-Judge vs human eval?
**A**: 

| | LLM-as-Judge | Human Eval |
|-|-------------|-----------|
| **Speed** | 1000 evals/minute | 10-50 evals/hour |
| **Cost** | ~$0.001/eval | ~$0.50-2/eval |
| **Consistency** | Very consistent | Inter-rater variance |
| **Nuance** | Misses edge cases | Catches subtlety |
| **Best for** | Regression testing, CI/CD | Periodic validation, launch decisions |

- **Hybrid**: LLM-judge cho mọi deployment (automated gate). Human eval monthly (calibration).
- **Calibrate**: run LLM-judge on 100 human-evaluated samples → measure correlation. r > 0.8 = good proxy.

### Q30: Agentic RAG evaluation?
**A**: Standard RAG metrics + agent-specific:
- **Retrieval iterations**: how many searches before finding answer?
- **Query refinement quality**: did refined queries improve results?
- **Tool selection accuracy**: did agent choose right retrieval tool?
- **Self-correction rate**: did agent catch its own bad retrievals?
- **Follow-up**: "Dataset cho evaluation?" → Use existing Q&A pairs from your docs + manually create 50-100 hard cases (multi-hop, ambiguous, unanswerable).

---

## Production Patterns (5 câu) 🆕

### Q31: Embedding model selection?
**A**: 

| Model | Dims | Cost | Quality (MTEB) | Best for |
|-------|:----:|:----:|:---------------:|----------|
| text-embedding-3-small | 1536 | $0.02/1M | Good | Budget, general |
| text-embedding-3-large | 3072 | $0.13/1M | Better | High quality |
| BGE-large-en-v1.5 | 1024 | Free (self-host) | Good | Privacy, no API |
| Cohere embed-v3 | 1024 | $0.10/1M | Best | Multilingual |
| Jina v3 | 1024 | $0.02/1M | Good | Long context (8K) |

- **Decision**: start with text-embedding-3-small. Switch to large/Cohere khi quality matters.

### Q32: Caching strategies cho RAG?
**A**: 3 levels:
1. **Query cache**: exact query → cached response (Redis, TTL 1-24h)
2. **Semantic cache**: similar query → cached response (embed query, check cosine > 0.95)
3. **Embedding cache**: document unchanged → reuse embedding (save recomputation)

```python
import hashlib, redis

def cached_rag(query: str, cache_ttl: int = 3600):
    cache_key = hashlib.sha256(query.encode()).hexdigest()
    
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)  # Cache hit
    
    result = rag_pipeline(query)   # Cache miss
    redis.setex(cache_key, cache_ttl, json.dumps(result))
    return result
```

### Q33: Multi-tenancy trong RAG?
**A**: Multiple customers sharing RAG infrastructure, data isolated.
- **Namespace isolation**: each tenant = separate collection/namespace in vector DB
- **Metadata filtering**: single collection, filter by `tenant_id` at query time
- **Separate DB**: each tenant = own vector DB instance (strongest isolation)
- **⚠️ Critical**: prompt leakage — tenant A's data must NEVER appear in tenant B's context. Test this explicitly.

### Q34: RAG pipeline observability?
**A**: Trace every retrieval step:
1. **Query**: original query + any expansions
2. **Retrieval**: chunks returned, scores, latency
3. **Reranking**: before/after order, score changes
4. **Generation**: prompt to LLM, response, tokens
5. **Feedback**: user rating, was answer helpful?

**Tools**: LangSmith (LangChain native), Langfuse (open-source), custom logging.

### Q35: Contextual retrieval (Anthropic's approach)?
**A**: 
- **Problem**: standard chunks lose document context when embedded independently
- **Solution**: use LLM to generate context for each chunk → prepend → embed enriched chunk
- **Result**: 49% reduction in retrieval failures, 67% with hybrid search
- **Cost**: 1 cheap LLM call per chunk (one-time during indexing)
- **When**: high-value knowledge base where retrieval quality directly impacts business
