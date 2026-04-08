"""
📐 Math for AI — Interactive Demo
Chạy: python math_demo.py

Demo: Linear Algebra, Probability, Statistics, Information Theory
Không cần API key hay external dependencies — chỉ numpy + built-in.
"""
import numpy as np
from collections import Counter
import math
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ════════════════════════════════════════════
# 1. LINEAR ALGEBRA
# ════════════════════════════════════════════

def demo_vectors():
    """Vectors, dot product, cosine similarity."""
    print("=" * 60)
    print("1. VECTORS & SIMILARITY")
    print("=" * 60)
    
    # Tạo "fake embeddings" cho demo
    embed_cat = np.array([0.9, 0.8, 0.1, 0.0, 0.2])
    embed_dog = np.array([0.8, 0.7, 0.2, 0.1, 0.3])
    embed_car = np.array([0.1, 0.0, 0.9, 0.8, 0.1])
    embed_bus = np.array([0.1, 0.0, 0.8, 0.9, 0.2])
    
    # Cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    pairs = [
        ("cat", "dog", embed_cat, embed_dog),
        ("cat", "car", embed_cat, embed_car),
        ("car", "bus", embed_car, embed_bus),
    ]
    
    print("\n  Cosine Similarity (semantic search basis):")
    for name1, name2, v1, v2 in pairs:
        sim = cosine_sim(v1, v2)
        bar = "█" * int(sim * 20)
        print(f"  {name1:>4} ↔ {name2:<4}: {sim:.3f} {bar}")
    
    # Euclidean distance
    print("\n  Euclidean Distance (lower = more similar):")
    for name1, name2, v1, v2 in pairs:
        dist = np.linalg.norm(v1 - v2)
        print(f"  {name1:>4} ↔ {name2:<4}: {dist:.3f}")

def demo_matrix_operations():
    """Matrix multiplication = Neural network forward pass."""
    print("\n" + "=" * 60)
    print("2. MATRIX MULTIPLICATION (Neural Network Layer)")
    print("=" * 60)
    
    # Simulate 1 neural network layer
    np.random.seed(42)
    
    X = np.array([[1.0, 0.5],    # 3 samples, 2 features
                   [0.0, 1.0],
                   [0.8, 0.2]])
    W = np.random.randn(2, 4) * 0.5   # 2 inputs → 4 neurons
    b = np.zeros(4)                     # bias
    
    # Forward pass: output = ReLU(X @ W + b)
    z = X @ W + b
    output = np.maximum(0, z)  # ReLU activation
    
    print(f"\n  Input X shape:  {X.shape}  (3 samples, 2 features)")
    print(f"  Weights W shape: {W.shape}  (2 inputs, 4 neurons)")
    print(f"  Output shape:   {output.shape}  (3 samples, 4 neurons)")
    print(f"\n  Rule: (m×n) @ (n×p) = (m×p)")
    print(f"  Neural net: every layer = matrix multiply + activation!")
    
    print(f"\n  Output (after ReLU):")
    for i, row in enumerate(output):
        print(f"    Sample {i}: [{', '.join(f'{v:.3f}' for v in row)}]")

def demo_svd():
    """SVD for dimensionality reduction."""
    print("\n" + "=" * 60)
    print("3. SVD (Singular Value Decomposition)")
    print("=" * 60)
    
    # Create a matrix with clear structure
    np.random.seed(42)
    # 2 "true" patterns + noise
    pattern1 = np.array([1, 0, 1, 0, 1])
    pattern2 = np.array([0, 1, 0, 1, 0])
    
    A = np.outer(np.random.randn(10), pattern1) * 3 + \
        np.outer(np.random.randn(10), pattern2) * 2 + \
        np.random.randn(10, 5) * 0.1  # Small noise
    
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    print(f"\n  Matrix shape: {A.shape}")
    print(f"\n  Singular values (importance of each pattern):")
    for i, s in enumerate(S):
        bar = "█" * int(s)
        print(f"    σ{i+1} = {s:6.2f} {bar}")
    
    # Reconstruction with top-k
    for k in [1, 2, 3]:
        A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        error = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
        print(f"\n  Rank-{k} approximation: {(1-error)*100:.1f}% information retained")

def demo_pca():
    """PCA = dimensionality reduction via eigenvalues."""
    print("\n" + "=" * 60)
    print("4. PCA (Principal Component Analysis)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate correlated data (5 features, but really 2 independent dimensions)
    n = 200
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)
    
    X = np.column_stack([
        z1 * 3,                    # Feature 1: mainly z1
        z1 * 2 + z2,              # Feature 2: mix
        z1 + z2 * 2,              # Feature 3: mix
        z2 * 3,                    # Feature 4: mainly z2
        z1 + z2 + np.random.randn(n) * 0.1  # Feature 5: noise
    ])
    
    # Manual PCA steps
    X_centered = X - X.mean(axis=0)
    cov_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    explained_var = eigenvalues / eigenvalues.sum()
    
    print(f"\n  Original: {X.shape[1]} features")
    print(f"\n  Explained variance per component:")
    cumulative = 0
    for i, (ev, var) in enumerate(zip(eigenvalues, explained_var)):
        cumulative += var
        bar = "█" * int(var * 40)
        print(f"    PC{i+1}: {var*100:5.1f}% (cumulative: {cumulative*100:.1f}%) {bar}")
    
    print(f"\n  → 2 components capture {sum(explained_var[:2])*100:.1f}% of variance!")
    print(f"  → Can reduce 5D → 2D with minimal info loss")


# ════════════════════════════════════════════
# 2. PROBABILITY & STATISTICS
# ════════════════════════════════════════════

def demo_probability():
    """Bayes theorem, distributions, softmax."""
    print("\n" + "=" * 60)
    print("5. PROBABILITY — Bayes Theorem")
    print("=" * 60)
    
    # Spam filter example
    p_spam = 0.3                    # Prior: 30% emails are spam
    p_free_given_spam = 0.8         # 80% spam contains "free"
    p_free_given_ham = 0.1          # 10% legitimate emails contain "free"
    
    # Bayes theorem
    p_free = p_free_given_spam * p_spam + p_free_given_ham * (1 - p_spam)
    p_spam_given_free = (p_free_given_spam * p_spam) / p_free
    
    print(f"\n  Prior P(spam) = {p_spam}")
    print(f"  Likelihood P('free'|spam) = {p_free_given_spam}")
    print(f"  Evidence P('free') = {p_free:.2f}")
    print(f"\n  ★ Posterior P(spam|'free') = {p_spam_given_free:.3f}")
    print(f"  → Email with 'free' has {p_spam_given_free*100:.1f}% chance of being spam")

def demo_softmax():
    """Softmax: logits → probabilities."""
    print("\n" + "=" * 60)
    print("6. SOFTMAX — Logits to Probabilities")
    print("=" * 60)
    
    def softmax(logits, temperature=1.0):
        scaled = logits / temperature
        exp = np.exp(scaled - np.max(scaled))
        return exp / exp.sum()
    
    logits = np.array([2.0, 1.0, 0.5, -1.0])
    classes = ["Positive", "Neutral", "Mixed", "Negative"]
    
    for temp in [0.5, 1.0, 2.0]:
        probs = softmax(logits, temperature=temp)
        print(f"\n  Temperature = {temp}:")
        for cls, p in zip(classes, probs):
            bar = "█" * int(p * 30)
            print(f"    {cls:>10}: {p:.3f} {bar}")
    
    print(f"\n  T=0.5 → confident (peaky)")
    print(f"  T=1.0 → standard")
    print(f"  T=2.0 → uncertain (flat)")

# ════════════════════════════════════════════
# 3. INFORMATION THEORY
# ════════════════════════════════════════════

def demo_information_theory():
    """Entropy, cross-entropy, KL divergence."""
    print("\n" + "=" * 60)
    print("7. INFORMATION THEORY")
    print("=" * 60)
    
    def entropy(probs):
        probs = np.array([p for p in probs if p > 0])
        return -np.sum(probs * np.log2(probs))
    
    def cross_entropy(p, q):
        p, q = np.array(p), np.clip(np.array(q), 1e-15, 1)
        return -np.sum(p * np.log2(q))
    
    def kl_divergence(p, q):
        return cross_entropy(p, q) - entropy(p)
    
    # Entropy = uncertainty
    print("\n  Entropy (uncertainty measure):")
    cases = [
        ("Fair coin [0.5, 0.5]", [0.5, 0.5]),
        ("Biased [0.9, 0.1]", [0.9, 0.1]),
        ("Certain [1.0, 0.0]", [1.0, 0.0]),
        ("Uniform 4-class", [0.25, 0.25, 0.25, 0.25]),
    ]
    for name, probs in cases:
        h = entropy(probs)
        bar = "█" * int(h * 10)
        print(f"    {name:30s}: {h:.3f} bits {bar}")
    
    # Cross-entropy = classification loss
    print("\n  Cross-Entropy Loss (classification):")
    true = [1, 0, 0]  # True class = 0
    predictions = [
        ("Good prediction [0.9, 0.05, 0.05]", [0.9, 0.05, 0.05]),
        ("OK prediction   [0.6, 0.2, 0.2]", [0.6, 0.2, 0.2]),
        ("Bad prediction  [0.1, 0.8, 0.1]", [0.1, 0.8, 0.1]),
    ]
    for name, pred in predictions:
        ce = cross_entropy(true, pred)
        print(f"    {name}: CE = {ce:.3f}")
    
    # KL Divergence
    print("\n  KL Divergence (distribution comparison):")
    p = [0.4, 0.3, 0.2, 0.1]
    q_uniform = [0.25, 0.25, 0.25, 0.25]
    q_similar = [0.35, 0.30, 0.20, 0.15]
    
    print(f"    P = {p}")
    print(f"    KL(P || Uniform): {kl_divergence(p, q_uniform):.4f}")
    print(f"    KL(P || Similar): {kl_divergence(p, q_similar):.4f}")
    print(f"    → Closer distribution = smaller KL ✅")


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

if __name__ == "__main__":
    print("🧮 MATH FOR AI — Interactive Demo")
    print("─" * 60)
    
    demo_vectors()
    demo_matrix_operations()
    demo_svd()
    demo_pca()
    demo_probability()
    demo_softmax()
    demo_information_theory()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("Key takeaways:")
    print("  • Cosine similarity = basis of semantic search")
    print("  • Matrix multiplication = every neural network layer")
    print("  • SVD/PCA = dimensionality reduction")
    print("  • Softmax temperature = control randomness")
    print("  • Cross-entropy = classification loss function")
    print("  • KL divergence = measure distribution difference")
    print("=" * 60)
