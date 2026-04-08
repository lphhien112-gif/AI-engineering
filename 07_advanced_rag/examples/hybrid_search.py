"""
🔎 Hybrid Search Demo — BM25 + Vector (simulated)
Chạy: python hybrid_search.py
"""
import re
import math
import numpy as np
from collections import Counter


# --- BM25 Implementation ---
class BM25:
    """BM25 ranking algorithm."""
    
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = documents
        self.tokenized = [self._tokenize(doc) for doc in documents]
        self.doc_count = len(documents)
        self.avgdl = sum(len(d) for d in self.tokenized) / self.doc_count
        
        # IDF
        self.idf = {}
        df = Counter()
        for doc in self.tokenized:
            for term in set(doc):
                df[term] += 1
        for term, freq in df.items():
            self.idf[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
    
    def score(self, query: str) -> np.ndarray:
        query_terms = self._tokenize(query)
        scores = np.zeros(self.doc_count)
        
        for term in query_terms:
            if term not in self.idf:
                continue
            for i, doc in enumerate(self.tokenized):
                tf = doc.count(term)
                dl = len(doc)
                score = self.idf[term] * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                scores[i] += score
        
        return scores
    
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())


# --- Simple Vector Search ---
class SimpleVectorSearch:
    """Simulated vector search using word overlap + TF-IDF-like embeddings."""
    
    def __init__(self, documents: list[str]):
        self.docs = documents
        # Build vocabulary
        all_words = set()
        for doc in documents:
            all_words.update(re.findall(r'\w+', doc.lower()))
        self.vocab = sorted(all_words)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        
        # Create document embeddings (TF-IDF-like)
        self.embeddings = np.zeros((len(documents), len(self.vocab)))
        for i, doc in enumerate(documents):
            words = re.findall(r'\w+', doc.lower())
            for word in words:
                if word in self.word2idx:
                    self.embeddings[i, self.word2idx[word]] += 1
        
        # L2 normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings = self.embeddings / norms
    
    def score(self, query: str) -> np.ndarray:
        # Create query embedding
        q_embed = np.zeros(len(self.vocab))
        for word in re.findall(r'\w+', query.lower()):
            if word in self.word2idx:
                q_embed[self.word2idx[word]] += 1
        
        norm = np.linalg.norm(q_embed)
        if norm > 0:
            q_embed = q_embed / norm
        
        # Cosine similarity
        return self.embeddings @ q_embed


# --- Hybrid Search ---
class HybridSearch:
    """Combine BM25 + Vector search."""
    
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.bm25 = BM25(documents)
        self.vector = SimpleVectorSearch(documents)
    
    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> list[dict]:
        bm25_scores = self.bm25.score(query)
        vector_scores = self.vector.score(query)
        
        # Normalize
        bm25_norm = self._normalize(bm25_scores)
        vector_norm = self._normalize(vector_scores)
        
        # Hybrid score
        hybrid = alpha * vector_norm + (1 - alpha) * bm25_norm
        
        # Rank
        indices = np.argsort(hybrid)[::-1][:top_k]
        
        return [
            {
                "rank": r + 1,
                "doc_id": int(i),
                "hybrid_score": round(hybrid[i], 4),
                "bm25_score": round(bm25_scores[i], 4),
                "vector_score": round(vector_scores[i], 4),
                "text": self.documents[i][:100] + "...",
            }
            for r, i in enumerate(indices)
            if hybrid[i] > 0
        ]
    
    @staticmethod
    def _normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple]:
    """RRF: Combine multiple rankings."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def main():
    print("=" * 60)
    print("🔎 Hybrid Search Demo")
    print("=" * 60)
    
    # Knowledge base
    documents = [
        "RLHF (Reinforcement Learning from Human Feedback) aligns language models with human preferences using reward models and PPO.",
        "The Transformer architecture uses self-attention to process sequences in parallel, replacing recurrent neural networks.",
        "Docker containers package applications with dependencies for consistent deployment across environments.",
        "Error code XJ-4021 indicates a timeout in the message queue service. Restart the consumer.",
        "LoRA (Low-Rank Adaptation) fine-tunes large models efficiently by training small rank decomposition matrices.",
        "Kubernetes orchestrates containerized applications across clusters with auto-scaling and self-healing.",
        "Vector databases like ChromaDB and Pinecone store embeddings for semantic similarity search.",
        "BM25 is a probabilistic ranking function based on term frequency and inverse document frequency.",
        "RAG (Retrieval Augmented Generation) combines search with LLMs to generate grounded answers.",
        "The attention mechanism computes Query, Key, Value matrices for contextual token representations.",
    ]
    
    searcher = HybridSearch(documents)
    
    # Test queries
    queries = [
        ("What is RLHF?", 0.7),           # Semantic → vector-heavy
        ("error code XJ-4021", 0.2),        # Exact match → BM25-heavy
        ("fine-tuning transformers", 0.5),   # Both needed → balanced
        ("docker kubernetes deploy", 0.4),   # Keywords → slightly BM25
    ]
    
    for query, alpha in queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: \"{query}\" (alpha={alpha})")
        print(f"   alpha={alpha}: {alpha*100:.0f}% vector + {(1-alpha)*100:.0f}% BM25")
        print(f"{'='*60}")
        
        results = searcher.search(query, top_k=3, alpha=alpha)
        for r in results:
            print(f"  #{r['rank']} [hybrid={r['hybrid_score']:.3f}] "
                  f"(BM25={r['bm25_score']:.2f}, Vec={r['vector_score']:.3f})")
            print(f"     {r['text']}")
    
    # RRF Demo
    print(f"\n{'='*60}")
    print("📊 Reciprocal Rank Fusion (RRF) Demo")
    print(f"{'='*60}")
    
    bm25_ranking = [3, 7, 0, 1, 4]
    vector_ranking = [0, 4, 1, 7, 3]
    
    print(f"  BM25 ranking:   {bm25_ranking}")
    print(f"  Vector ranking: {vector_ranking}")
    
    fused = reciprocal_rank_fusion([bm25_ranking, vector_ranking])
    print(f"  RRF result:")
    for doc_id, score in fused[:5]:
        print(f"    Doc {doc_id}: RRF score = {score:.4f} — \"{documents[doc_id][:60]}...\"")
    
    print(f"\n{'='*60}")
    print("✅ Hybrid Search Demo completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
