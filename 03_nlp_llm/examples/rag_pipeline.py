"""
🔍 Simple RAG Pipeline — Chạy hoàn toàn local (không cần API key)
Chạy: pip install sentence-transformers chromadb
       python rag_pipeline.py
"""
import chromadb
from sentence_transformers import SentenceTransformer


class SimpleRAG:
    """RAG pipeline sử dụng local embedding model + ChromaDB."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="rag_demo",
            metadata={"hnsw:space": "cosine"},
        )
        self._doc_count = 0
    
    def add_documents(self, documents: list[str], metadatas: list[dict] = None):
        """Add documents to the vector store."""
        embeddings = self.embedder.encode(documents).tolist()
        ids = [f"doc_{self._doc_count + i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        self._doc_count += len(documents)
        print(f"  ✅ Added {len(documents)} documents (total: {self._doc_count})")
    
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve most relevant documents for a query."""
        query_embedding = self.embedder.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._doc_count),
        )
        
        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return retrieved
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Full RAG pipeline: retrieve + generate (simulated)."""
        # 1. Retrieve relevant documents
        docs = self.retrieve(question, top_k)
        
        # 2. Format context
        context = "\n---\n".join([d["content"] for d in docs])
        
        # 3. Generate answer (simulated — replace with LLM call in production)
        prompt = f"""Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:"""
        
        # In production: call OpenAI/Gemini API here
        return prompt, docs


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Simple recursive chunking by characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text[start:end].rfind(". ")
            if last_period > chunk_size // 2:
                end = start + last_period + 2
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 20]


def main():
    print("=" * 60)
    print("🔍 Simple RAG Pipeline Demo")
    print("=" * 60)
    
    # 1. Create RAG instance
    rag = SimpleRAG()
    
    # 2. Knowledge base
    knowledge = [
        "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "Machine Learning is a subset of artificial intelligence that focuses on building systems that learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
        "Deep Learning is a subset of machine learning using neural networks with many layers. Popular architectures include CNNs for images, RNNs/LSTMs for sequences, and Transformers for NLP.",
        "Transformers, introduced in the 'Attention is All You Need' paper (2017), use self-attention mechanisms to process sequences in parallel. They are the foundation of modern LLMs like GPT-4, Claude, and Gemini.",
        "RAG (Retrieval-Augmented Generation) combines a retrieval system with a generative model. It retrieves relevant documents from a knowledge base and uses them as context for generating accurate answers.",
        "Vector databases store high-dimensional vectors (embeddings) and enable efficient similarity search. Popular choices include Qdrant, ChromaDB, pgvector, and Pinecone.",
        "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method. It adds small trainable matrices to frozen model weights, reducing VRAM requirements by 10-100x.",
        "FastAPI is a modern Python web framework for building APIs. It provides automatic validation, documentation, async support, and high performance comparable to NodeJS.",
        "Docker containers package applications with their dependencies for consistent deployment. Multi-stage builds reduce image size by separating build and runtime stages.",
        "MLOps combines ML and DevOps practices for production ML systems. Key components include CI/CD, model registry, monitoring, data versioning, and experiment tracking.",
    ]
    
    # 3. Index documents
    print("\n📥 Indexing documents...")
    metadatas = [
        {"topic": "python"}, {"topic": "ml"}, {"topic": "dl"},
        {"topic": "transformers"}, {"topic": "rag"}, {"topic": "vectordb"},
        {"topic": "finetuning"}, {"topic": "api"}, {"topic": "docker"},
        {"topic": "mlops"},
    ]
    rag.add_documents(knowledge, metadatas)
    
    # 4. Test queries
    queries = [
        "What is RAG and how does it work?",
        "How do Transformers differ from RNNs?",
        "What is LoRA fine-tuning?",
        "How to deploy ML models?",
    ]
    
    print("\n🔍 Running queries...\n")
    for query in queries:
        print(f"❓ Query: {query}")
        docs = rag.retrieve(query, top_k=2)
        for i, doc in enumerate(docs, 1):
            distance = doc.get("distance", "N/A")
            print(f"  📄 [{doc['metadata']['topic']}] (distance: {distance:.4f})")
            print(f"     {doc['content'][:100]}...")
        print()
    
    # 5. Chunking demo
    print("=" * 60)
    print("📝 Chunking Demo")
    print("=" * 60)
    
    long_text = """
    Machine learning is revolutionizing every industry. In healthcare, it's used for 
    disease diagnosis and drug discovery. In finance, it powers fraud detection and 
    algorithmic trading. In manufacturing, it enables predictive maintenance and quality 
    control. The key to successful ML projects is high-quality data. Data scientists 
    spend 80% of their time on data preparation, including cleaning, feature engineering, 
    and validation. Modern approaches emphasize data-centric AI, where improving data 
    quality is prioritized over model architecture changes.
    """
    
    chunks = chunk_text(long_text, chunk_size=150, overlap=30)
    print(f"\nOriginal text: {len(long_text)} chars")
    print(f"Chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: ({len(chunk)} chars) {chunk[:80]}...")
    
    print("\n" + "=" * 60)
    print("✅ RAG Pipeline Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
