# 🗄️ Vector Databases — Production Guide

> **Mục tiêu**: HNSW, pgvector, Qdrant, ChromaDB, Pinecone — từ prototype đến production.
> Vector DB = foundation of every RAG system and semantic search.

---

## 1. Vector DB trong AI Pipeline

```mermaid
graph LR
    A[Documents] -->|chunk| B[Chunks]
    B -->|embed| C[Vectors<br/>384-3072 dim]
    C -->|store| D[(Vector DB)]
    
    E[User Query] -->|embed| F[Query Vector]
    F -->|search| D
    D -->|top-K| G[Relevant Chunks]
    G --> H[LLM Context]
    H --> I[Answer]
    
    style D fill:#e1f5fe
```

```
Traditional DB: SELECT * FROM docs WHERE title = "AI"     → exact match
Vector DB:      Find docs SIMILAR TO embed("what is AI?")  → semantic match
```

---

## 2. ANN Algorithms — How Vector Search Works

### 2.1 HNSW (Hierarchical Navigable Small World)

```mermaid
graph TB
    subgraph "Layer 2 (sparse, fast navigation)"
        L2A((A)) --- L2D((D))
    end

    subgraph "Layer 1 (medium)"
        L1A((A)) --- L1B((B))
        L1A --- L1D((D))
        L1B --- L1D
    end

    subgraph "Layer 0 (dense, accurate)"
        L0A((A)) --- L0B((B))
        L0A --- L0C((C))
        L0B --- L0C
        L0B --- L0D((D))
        L0C --- L0D
        L0C --- L0E((E))
        L0D --- L0E
    end
    
    L2A -.-> L1A -.-> L0A
    L2D -.-> L1D -.-> L0D
```

```python
# HNSW Intuition:
# 1. Start at top layer (few nodes, long-range connections)
# 2. Greedily navigate to nearest neighbor at each layer
# 3. Drop to next layer (more nodes, shorter connections)
# 4. Repeat until bottom layer → most accurate result

# Key parameters:
# M = max connections per node (16-64, higher = more accurate but more memory)
# ef_construction = beam width during build (200-500, higher = better graph)
# ef_search = beam width during query (50-200, higher = more accurate)
```

### 2.2 ANN Algorithm Comparison

| Algorithm | Speed | Accuracy | Memory | Build Speed | Best For |
|-----------|:-----:|:--------:|:------:|:-----------:|---------|
| **HNSW** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High | Slow | General purpose (default!) |
| **IVF** | ⭐⭐⭐⭐ | ⭐⭐⭐ | Medium | Fast | Large scale, GPU |
| **IVF-PQ** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Low | Fast | Billions of vectors |
| **Flat** | ⭐ | ⭐⭐⭐⭐⭐ | Medium | Instant | <10K vectors (exact) |
| **ScaNN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Fast | Google-scale search |

---

## 3. ChromaDB (Dev & Prototyping)

```python
import chromadb
from chromadb.utils import embedding_functions

# ── Setup ──
client = chromadb.PersistentClient(path="./chroma_db")

# Use built-in embedding function (sentence-transformers)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine", "hnsw:M": 32},
)

# ── Add documents (auto-embeds!) ──
docs = [
    "Machine learning is a subset of AI that learns from data",
    "Deep learning uses neural networks with many layers",
    "Natural language processing handles text and speech",
    "Computer vision processes images and video",
    "Reinforcement learning learns through trial and error",
]

collection.add(
    documents=docs,
    metadatas=[
        {"topic": "ml", "level": "basic"},
        {"topic": "dl", "level": "basic"},
        {"topic": "nlp", "level": "basic"},
        {"topic": "cv", "level": "basic"},
        {"topic": "rl", "level": "basic"},
    ],
    ids=[f"doc_{i}" for i in range(len(docs))],
)

# ── Query (auto-embeds query!) ──
results = collection.query(
    query_texts=["How do neural networks work?"],
    n_results=3,
    where={"topic": {"$in": ["ml", "dl"]}},         # Metadata filter
    where_document={"$contains": "neural"},           # Content filter
)

for doc, score in zip(results["documents"][0], results["distances"][0]):
    print(f"  [{1-score:.3f}] {doc}")

# ── Update ──
collection.update(ids=["doc_0"], metadatas=[{"topic": "ml", "level": "advanced"}])

# ── Delete ──
collection.delete(where={"topic": "rl"})

print(f"Collection size: {collection.count()}")
```

---

## 4. Qdrant (Production-Ready)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    models,
)

# ── Connect ──
client = QdrantClient(host="localhost", port=6333)
# Or cloud: QdrantClient(url="https://xxx.qdrant.io", api_key="key")

# ── Create collection ──
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    # Optimizers
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=20000,      # Build HNSW after this many points
        memmap_threshold=50000,        # Use mmap for large collections
    ),
    # HNSW config
    hnsw_config=models.HnswConfigDiff(
        m=16,                          # Connections per node
        ef_construct=100,              # Build quality
    ),
)

# ── Upsert with rich payloads ──
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    {"text": "RAG retrieves relevant context for LLM generation", "topic": "rag", "year": 2024},
    {"text": "HNSW is the most popular ANN algorithm", "topic": "search", "year": 2023},
    {"text": "ChromaDB is great for prototyping", "topic": "db", "year": 2024},
]

embeddings = model.encode([d["text"] for d in documents])

client.upsert(
    collection_name="documents",
    points=[
        PointStruct(id=i, vector=emb.tolist(), payload=doc)
        for i, (emb, doc) in enumerate(zip(embeddings, documents))
    ],
)

# ── Advanced search with filtering ──
query_vector = model.encode("How does similarity search work?").tolist()

results = client.search(
    collection_name="documents",
    query_vector=query_vector,
    limit=5,
    query_filter=Filter(
        must=[
            FieldCondition(key="year", range=Range(gte=2024)),
            FieldCondition(key="topic", match=MatchValue(value="rag")),
        ]
    ),
    score_threshold=0.5,    # Min similarity score
)

for r in results:
    print(f"  [{r.score:.3f}] {r.payload['text']}")
```

---

## 5. pgvector (PostgreSQL Extension)

```sql
-- ══════════════════════════════════
-- Setup
-- ══════════════════════════════════
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    topic VARCHAR(50),
    source VARCHAR(200),
    created_at TIMESTAMP DEFAULT NOW(),
    embedding vector(384)          -- 384-dim vector
);

-- ══════════════════════════════════
-- Indexes (CRITICAL for performance!)
-- ══════════════════════════════════

-- HNSW index (recommended, fast search)
CREATE INDEX ON documents 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- IVFFlat index (alternative, faster build)
CREATE INDEX ON documents 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);  -- √n lists recommended

-- ══════════════════════════════════
-- Insert
-- ══════════════════════════════════
INSERT INTO documents (content, topic, embedding) VALUES
    ('RAG retrieves relevant context', 'rag', '[0.1, 0.2, ...]');

-- ══════════════════════════════════
-- Semantic Search (cosine similarity)
-- ══════════════════════════════════
SELECT 
    content,
    topic,
    1 - (embedding <=> $1::vector) AS similarity    -- $1 = query vector
FROM documents
WHERE topic = 'rag'                                  -- Pre-filter!
ORDER BY embedding <=> $1::vector                    -- Cosine distance
LIMIT 5;

-- ══════════════════════════════════
-- Hybrid Search (vector + full-text)
-- ══════════════════════════════════
-- Add full-text search index
ALTER TABLE documents ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX ON documents USING gin(search_vector);

-- Combine vector similarity + keyword matching
SELECT 
    content,
    (0.7 * (1 - (embedding <=> $1::vector))) +      -- Vector score (70%)
    (0.3 * ts_rank(search_vector, to_tsquery($2)))   -- BM25 score (30%)
    AS hybrid_score
FROM documents
WHERE search_vector @@ to_tsquery($2)                -- Keyword filter
ORDER BY hybrid_score DESC
LIMIT 10;
```

### pgvector with Python

```python
import psycopg2
import numpy as np

conn = psycopg2.connect("postgresql://user:pass@localhost/dbname")
cur = conn.cursor()

# Register pgvector type
from pgvector.psycopg2 import register_vector
register_vector(conn)

# Insert
embedding = model.encode("RAG is powerful for QA systems")
cur.execute(
    "INSERT INTO documents (content, topic, embedding) VALUES (%s, %s, %s)",
    ("RAG is powerful for QA systems", "rag", embedding)
)

# Search
query_embedding = model.encode("How does RAG work?")
cur.execute("""
    SELECT content, 1 - (embedding <=> %s) AS similarity
    FROM documents
    ORDER BY embedding <=> %s
    LIMIT 5
""", (query_embedding, query_embedding))

for row in cur.fetchall():
    print(f"  [{row[1]:.3f}] {row[0]}")
```

---

## 6. Scaling & Production Tips

```mermaid
graph TB
    subgraph "Small (<100K vectors)"
        A["ChromaDB (in-process)<br/>or pgvector"]
    end
    
    subgraph "Medium (100K-10M)"
        B["pgvector (HNSW)<br/>or Qdrant"]
    end
    
    subgraph "Large (10M-1B)"
        C["Qdrant cluster<br/>or Pinecone"]
    end
    
    subgraph "Massive (>1B)"
        D["Milvus + GPU<br/>or custom sharding"]
    end
```

```
Production checklist:
✅ HNSW index (not flat/brute-force)
✅ Pre-filter BEFORE vector search (topic, date range)
✅ Batch inserts (1000+ at a time, not one-by-one)
✅ Connection pooling (pgvector)
✅ Monitor: query latency, index size, recall
✅ Cache frequent queries (Redis)
✅ Quantization for large scale (reduce 4x memory)
```

---

## 7. Decision Guide

| Situation | Recommendation | Why |
|-----------|---------------|-----|
| **Prototyping** | ChromaDB | Zero setup, in-process |
| **Have PostgreSQL** | pgvector | One database, SQL power |
| **Need max performance** | Qdrant | Rust-fast, advanced filtering |
| **Zero maintenance** | Pinecone | Fully managed, serverless |
| **Multi-modal** | Weaviate | Built-in vectorizers |
| **Billion-scale** | Milvus | GPU support, distributed |
| **Supabase stack** | pgvector | Native integration |

---

## 🎯 Interview Tips — Chi Tiết

### Q1: "HNSW hoạt động thế nào?"
**A**: Multi-layer skip-list graph. Top layer: sparse (fast navigation). Bottom: dense (accurate). Search: start top → greedy navigate → drop down → repeat. O(log n) search time. Best overall ANN algorithm.

### Q2: "Exact search vs ANN?"
**A**: Exact: O(n), 100% recall, brute-force. ANN: O(log n), ~95-99% recall. Production ALWAYS uses ANN. HNSW recall@10 > 99% with proper tuning (ef_search=100+).

### Q3: "pgvector vs Qdrant?"
**A**: pgvector: if already have PostgreSQL, SQL power, hybrid search easy. Qdrant: if need max throughput, advanced filtering, built for scale. pgvector for 80% use cases, Qdrant when pgvector is bottleneck.

### Q4: "Embedding dimension trade-off?"
**A**: Higher dim = more information, better accuracy, but more storage + slower search. 384 (MiniLM) → fast, good for prototyping. 1024 (bge-large) → production quality. 3072 (text-embedding-3-large) → best but expensive.

### Q5: "Hybrid search?"
**A**: Combine vector (semantic) + BM25 (keyword). Retrieve candidates from both, merge with RRF or weighted scoring. Catches cases where semantic misses exact keywords and vice versa. Standard in production RAG.

### Q6: "How to scale vector DB?"
**A**: (1) HNSW index tuning (M, ef). (2) Pre-filter before search. (3) Quantization (4x memory reduction). (4) Sharding for >10M vectors. (5) Cache frequent queries. (6) Separate read/write replicas.
