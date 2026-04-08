"""
✂️ Chunking Strategies Demo — Compare different methods
Chạy: python chunking_demo.py
"""
import re
import hashlib


SAMPLE_DOCUMENT = """
# Introduction to Transformer Architecture

The Transformer architecture was introduced in the paper "Attention Is All You Need" 
by Vaswani et al. in 2017. It revolutionized natural language processing by replacing 
recurrent neural networks with self-attention mechanisms.

## Self-Attention Mechanism

Self-attention allows each position in the sequence to attend to all other positions. 
The mechanism computes Query, Key, and Value matrices from the input embeddings. 
The attention score is calculated as the scaled dot product of Query and Key matrices, 
followed by a softmax operation. This produces attention weights that are applied to 
the Value matrix.

The multi-head attention variant runs several attention operations in parallel, 
allowing the model to attend to information from different representation subspaces. 
Each head captures different types of relationships between tokens.

## Positional Encoding

Since Transformers don't have inherent notion of sequence order, positional encodings 
are added to the input embeddings. The original paper used sinusoidal functions:
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Modern alternatives include learned positional embeddings, RoPE (Rotary Position 
Embedding), and ALiBi (Attention with Linear Biases).

## Encoder-Decoder Structure

The original Transformer has an encoder-decoder structure. The encoder processes the 
input sequence bidirectionally. The decoder generates output tokens autoregressively, 
attending to both the encoder output (cross-attention) and previous decoder outputs 
(masked self-attention).

Variants:
- Encoder-only: BERT, RoBERTa (for classification, NER)
- Decoder-only: GPT, LLaMA (for generation)
- Encoder-decoder: T5, BART (for seq2seq tasks)

## Scaling Laws

Research by Kaplan et al. showed that Transformer performance scales predictably with 
model size, dataset size, and compute budget. The Chinchilla paper further refined 
these scaling laws, showing that many models were undertrained relative to their size.
"""


def fixed_size_chunk(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Naive fixed-size chunking."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def sentence_chunk(text: str, max_sentences: int = 3) -> list[str]:
    """Chunk by sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks


def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Recursive character splitting (like LangChain)."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def _split(text, separators, chunk_size):
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current = ""
                
                for part in parts:
                    if len(current) + len(part) + len(sep) <= chunk_size:
                        current = current + sep + part if current else part
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = part
                
                if current:
                    chunks.append(current.strip())
                
                if all(len(c) <= chunk_size for c in chunks):
                    return [c for c in chunks if c]
        
        # Fallback: hard split
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    
    return _split(text, separators, chunk_size)


def markdown_header_chunk(text: str) -> list[dict]:
    """Chunk by markdown headers (document-aware)."""
    chunks = []
    current_header = "Introduction"
    current_content = []
    
    for line in text.split("\n"):
        if line.startswith("## "):
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    chunks.append({
                        "header": current_header,
                        "content": content,
                    })
            current_header = line.replace("## ", "").strip()
            current_content = []
        elif line.startswith("# "):
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    chunks.append({
                        "header": current_header,
                        "content": content,
                    })
            current_header = line.replace("# ", "").strip()
            current_content = []
        else:
            current_content.append(line)
    
    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            chunks.append({"header": current_header, "content": content})
    
    return chunks


def semantic_chunk_simulation(text: str) -> list[str]:
    """Simulated semantic chunking (uses text similarity heuristic)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Simple heuristic: check word overlap as proxy for semantic similarity
        prev_words = set(sentences[i-1].lower().split())
        curr_words = set(sentences[i].lower().split())
        
        overlap = len(prev_words & curr_words) / max(len(prev_words | curr_words), 1)
        
        if overlap < 0.15 and len(" ".join(current_chunk)) > 100:
            # Low similarity → new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def main():
    print("=" * 60)
    print("✂️ Chunking Strategies Comparison")
    print("=" * 60)
    
    doc = SAMPLE_DOCUMENT.strip()
    print(f"\n📄 Document: {len(doc)} characters, {len(doc.split())} words\n")
    
    # 1. Fixed-size
    print("=== 1. Fixed-Size Chunking (300 chars, 50 overlap) ===")
    chunks = fixed_size_chunk(doc, chunk_size=300, overlap=50)
    print(f"  Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  [{i}] ({len(chunk)} chars): \"{chunk[:80]}...\"")
    print(f"  ⚠️  Problem: Cuts mid-sentence, loses semantic coherence\n")
    
    # 2. Sentence-based
    print("=== 2. Sentence Chunking (3 sentences per chunk) ===")
    chunks = sentence_chunk(doc, max_sentences=3)
    print(f"  Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  [{i}] ({len(chunk)} chars): \"{chunk[:80]}...\"")
    print(f"  ✅ Better: Complete sentences preserved\n")
    
    # 3. Recursive
    print("=== 3. Recursive Character Splitting (500 chars) ===")
    chunks = recursive_chunk(doc, chunk_size=500)
    print(f"  Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  [{i}] ({len(chunk)} chars): \"{chunk[:80]}...\"")
    print(f"  ✅ Good: Tries paragraph → sentence → word boundaries\n")
    
    # 4. Markdown headers
    print("=== 4. Markdown Header Chunking (document-aware) ===")
    chunks = markdown_header_chunk(doc)
    print(f"  Chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"  [{chunk['header']}] ({len(chunk['content'])} chars): "
              f"\"{chunk['content'][:60]}...\"")
    print(f"  ✅ Best: Respects document structure, includes metadata\n")
    
    # 5. Semantic (simulated)
    print("=== 5. Semantic Chunking (simulated) ===")
    chunks = semantic_chunk_simulation(doc)
    print(f"  Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  [{i}] ({len(chunk)} chars): \"{chunk[:80]}...\"")
    print(f"  ✅ Best quality: Coherent semantic units (needs embeddings in real use)\n")
    
    # Summary comparison
    print("=" * 60)
    print("📊 Summary Comparison")
    print("=" * 60)
    
    methods = [
        ("Fixed-size", len(fixed_size_chunk(doc, 300, 50)), "Fast", "Cuts mid-sentence"),
        ("Sentence", len(sentence_chunk(doc, 3)), "Fast", "Fixed grouping"),
        ("Recursive", len(recursive_chunk(doc, 500)), "Fast", "May miss semantics"),
        ("Markdown", len(markdown_header_chunk(doc)), "Fast", "Only works for structured docs"),
        ("Semantic", len(semantic_chunk_simulation(doc)), "Slow", "Needs embeddings model"),
    ]
    
    print(f"  {'Method':<16} {'Chunks':<8} {'Speed':<8} {'Limitation'}")
    print(f"  {'-'*50}")
    for name, n_chunks, speed, limitation in methods:
        print(f"  {name:<16} {n_chunks:<8} {speed:<8} {limitation}")
    
    print(f"\n{'='*60}")
    print("✅ Chunking Demo completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
