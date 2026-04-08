"""
🔢 Embeddings Demo — Generate, compare, and search with embeddings
Chạy: pip install sentence-transformers scikit-learn numpy
       python embeddings_demo.py
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    print("=" * 60)
    print("🔢 Embeddings Demo")
    print("=" * 60)
    
    # 1. Load model (runs locally, no API key needed)
    print("\n📥 Loading model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # 2. Generate embeddings
    sentences = [
        "Machine learning is a branch of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing handles text data",
        "Computer vision processes images and videos",
        "I love eating pizza on Friday nights",
        "The stock market crashed yesterday",
    ]
    
    print("\n📊 Generating embeddings...")
    embeddings = model.encode(sentences)
    print(f"   Shape: {embeddings.shape}")  # (6, 384)
    
    # 3. Similarity matrix
    print("\n🔗 Cosine Similarity Matrix:")
    sim_matrix = cosine_similarity(embeddings)
    
    # Print header
    labels = ["ML", "DL", "NLP", "CV", "Pizza", "Stock"]
    print(f"{'':>8}", end="")
    for l in labels:
        print(f"{l:>8}", end="")
    print()
    
    for i, label in enumerate(labels):
        print(f"{label:>8}", end="")
        for j in range(len(labels)):
            score = sim_matrix[i][j]
            print(f"{score:>8.3f}", end="")
        print()
    
    # 4. Semantic search
    print("\n🔍 Semantic Search Demo:")
    queries = [
        "How do neural networks learn?",
        "What is AI?",
        "Best food for dinner",
    ]
    
    for query in queries:
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_idx = np.argsort(similarities)[::-1][:3]
        
        print(f"\n  Query: '{query}'")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    #{rank} ({similarities[idx]:.3f}): {sentences[idx]}")
    
    # 5. Clustering
    print("\n📦 Simple Clustering (by similarity threshold):")
    threshold = 0.5
    clusters = []
    assigned = set()
    
    for i in range(len(sentences)):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, len(sentences)):
            if j not in assigned and sim_matrix[i][j] > threshold:
                cluster.append(j)
                assigned.add(j)
        clusters.append(cluster)
    
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i + 1}:")
        for idx in cluster:
            print(f"    - {sentences[idx]}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
