"""
📏 RAG Evaluation Demo — RAGAS-style evaluation
Chạy: python rag_evaluation.py
"""
import re
import numpy as np


class RAGEvaluator:
    """Evaluate RAG system quality with RAGAS-style metrics."""
    
    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Check if answer is grounded in context (no hallucination)."""
        answer_words = set(re.findall(r'\w+', answer.lower()))
        context_words = set()
        for ctx in contexts:
            context_words.update(re.findall(r'\w+', ctx.lower()))
        
        if not answer_words:
            return 0.0
        
        grounded = len(answer_words & context_words) / len(answer_words)
        return round(min(grounded * 1.5, 1.0), 3)  # Scale up, cap at 1.0
    
    def answer_relevancy(self, question: str, answer: str) -> float:
        """Check if answer addresses the question."""
        q_words = set(re.findall(r'\w+', question.lower()))
        a_words = set(re.findall(r'\w+', answer.lower()))
        
        # Remove stopwords
        stopwords = {'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do', 'are', 'was', 'in', 'of', 'to', 'and', 'for'}
        q_content = q_words - stopwords
        
        if not q_content:
            return 1.0
        
        overlap = len(q_content & a_words) / len(q_content)
        length_penalty = min(len(a_words) / 10, 1.0)  # Penalize very short answers
        
        return round(overlap * 0.6 + length_penalty * 0.4, 3)
    
    def context_precision(self, question: str, contexts: list[str]) -> float:
        """Are retrieved contexts relevant to the question?"""
        q_words = set(re.findall(r'\w+', question.lower()))
        stopwords = {'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do', 'are', 'was', 'in', 'of', 'to', 'and', 'for'}
        q_content = q_words - stopwords
        
        if not q_content or not contexts:
            return 0.0
        
        relevant = 0
        for ctx in contexts:
            ctx_words = set(re.findall(r'\w+', ctx.lower()))
            overlap = len(q_content & ctx_words) / len(q_content)
            if overlap > 0.2:
                relevant += 1
        
        return round(relevant / len(contexts), 3)
    
    def context_recall(self, answer: str, contexts: list[str], ground_truth: str) -> float:
        """Do contexts contain info needed for the ground truth answer?"""
        gt_words = set(re.findall(r'\w+', ground_truth.lower()))
        stopwords = {'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do', 'are', 'was', 'in', 'of', 'to', 'and', 'for'}
        gt_content = gt_words - stopwords
        
        if not gt_content:
            return 1.0
        
        context_words = set()
        for ctx in contexts:
            context_words.update(re.findall(r'\w+', ctx.lower()))
        
        recall = len(gt_content & context_words) / len(gt_content)
        return round(recall, 3)
    
    def answer_correctness(self, answer: str, ground_truth: str) -> float:
        """How correct is the answer compared to ground truth?"""
        a_words = set(re.findall(r'\w+', answer.lower()))
        gt_words = set(re.findall(r'\w+', ground_truth.lower()))
        
        if not gt_words:
            return 0.0
        
        precision = len(a_words & gt_words) / len(a_words) if a_words else 0
        recall = len(a_words & gt_words) / len(gt_words)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return round(f1, 3)


def retrieval_metrics(retrieved_ids: list[int], relevant_ids: list[int], k: int = 5):
    """Standard retrieval evaluation metrics."""
    retrieved = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    # Precision@K
    hits = sum(1 for doc in retrieved if doc in relevant_set)
    precision_at_k = hits / k
    
    # Recall
    recall = hits / len(relevant_set) if relevant_set else 0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0
    for i, doc in enumerate(retrieved):
        if doc in relevant_set:
            mrr = 1.0 / (i + 1)
            break
    
    # NDCG
    dcg = sum(1.0 / np.log2(i + 2) for i, doc in enumerate(retrieved) if doc in relevant_set)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    return {
        f"precision@{k}": round(precision_at_k, 3),
        "recall": round(recall, 3),
        "mrr": round(mrr, 3),
        "ndcg": round(ndcg, 3),
    }


def main():
    print("=" * 60)
    print("📏 RAG Evaluation Demo")
    print("=" * 60)
    
    evaluator = RAGEvaluator()
    
    # Test cases
    test_cases = [
        {
            "question": "What is RLHF and how does it work?",
            "contexts": [
                "RLHF stands for Reinforcement Learning from Human Feedback. It trains a reward model from human preferences.",
                "The process involves collecting human comparisons of model outputs, training a reward model, then using PPO to optimize.",
            ],
            "answer": "RLHF (Reinforcement Learning from Human Feedback) is a technique that aligns language models with human preferences. It works by training a reward model from human comparisons and using PPO optimization.",
            "ground_truth": "RLHF uses human feedback to train a reward model, then uses reinforcement learning (PPO) to align the language model with human preferences.",
        },
        {
            "question": "How to deploy models with Docker?",
            "contexts": [
                "Python is a versatile programming language used in data science.",
                "Machine learning models can be trained on GPUs for faster training.",
            ],
            "answer": "Docker packages applications into containers for consistent deployment across environments.",
            "ground_truth": "Use Docker to containerize ML models with dependencies, create a Dockerfile, build the image, and deploy to container orchestration platforms.",
        },
        {
            "question": "What is the Transformer attention mechanism?",
            "contexts": [
                "The attention mechanism computes Query, Key, Value matrices. Attention scores are calculated as scaled dot products.",
                "Multi-head attention runs parallel attention operations to capture different relationships.",
                "Transformers replaced RNNs for sequence processing tasks.",
            ],
            "answer": "The Transformer attention mechanism computes Query, Key, and Value matrices from input embeddings. Attention scores are the scaled dot product of Q and K, applied to V. Multi-head attention runs multiple attention heads in parallel.",
            "ground_truth": "Transformer attention uses Query, Key, Value matrices with scaled dot-product attention and multi-head parallelism.",
        },
    ]
    
    print("\n=== 1. RAGAS-style Evaluation ===\n")
    
    all_scores = []
    for i, case in enumerate(test_cases, 1):
        faith = evaluator.faithfulness(case["answer"], case["contexts"])
        relevancy = evaluator.answer_relevancy(case["question"], case["answer"])
        precision = evaluator.context_precision(case["question"], case["contexts"])
        recall = evaluator.context_recall(case["answer"], case["contexts"], case["ground_truth"])
        correctness = evaluator.answer_correctness(case["answer"], case["ground_truth"])
        
        scores = {
            "faithfulness": faith,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "context_recall": recall,
            "answer_correctness": correctness,
        }
        all_scores.append(scores)
        
        print(f"  📋 Test Case {i}: \"{case['question'][:50]}...\"")
        for metric, value in scores.items():
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            status = "✅" if value > 0.7 else "⚠️" if value > 0.4 else "❌"
            print(f"     {status} {metric:<22} {bar} {value:.3f}")
        print()
    
    # Aggregate scores
    print("  📊 Aggregate Scores:")
    for metric in all_scores[0].keys():
        avg = np.mean([s[metric] for s in all_scores])
        print(f"     {metric:<22} {avg:.3f}")
    
    # 2. Retrieval metrics
    print(f"\n=== 2. Retrieval Metrics ===\n")
    
    scenarios = [
        ("Perfect retrieval", [1, 2, 3, 4, 5], [1, 2, 3]),
        ("Good retrieval", [1, 4, 2, 7, 8], [1, 2, 3]),
        ("Poor retrieval", [7, 8, 9, 4, 1], [1, 2, 3]),
        ("Miss completely", [6, 7, 8, 9, 10], [1, 2, 3]),
    ]
    
    print(f"  {'Scenario':<22} {'P@5':<8} {'Recall':<8} {'MRR':<8} {'NDCG':<8}")
    print(f"  {'-'*56}")
    
    for name, retrieved, relevant in scenarios:
        metrics = retrieval_metrics(retrieved, relevant, k=5)
        print(f"  {name:<22} {metrics['precision@5']:<8.3f} "
              f"{metrics['recall']:<8.3f} {metrics['mrr']:<8.3f} {metrics['ndcg']:<8.3f}")
    
    # 3. Regression test
    print(f"\n=== 3. Regression Testing ===\n")
    
    versions = [
        ("v1.0 (baseline)", {"faithfulness": 0.72, "relevancy": 0.68, "precision": 0.65}),
        ("v1.1 (new chunking)", {"faithfulness": 0.78, "relevancy": 0.75, "precision": 0.72}),
        ("v1.2 (+ reranking)", {"faithfulness": 0.85, "relevancy": 0.82, "precision": 0.80}),
        ("v1.3 (+ hybrid)", {"faithfulness": 0.88, "relevancy": 0.85, "precision": 0.83}),
    ]
    
    print(f"  {'Version':<24} {'Faith':<10} {'Rel':<10} {'Prec':<10} {'Δ Faith'}")
    print(f"  {'-'*60}")
    
    prev_faith = 0
    for name, metrics in versions:
        delta = metrics['faithfulness'] - prev_faith
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        arrow = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
        
        print(f"  {name:<24} {metrics['faithfulness']:<10.2f} "
              f"{metrics['relevancy']:<10.2f} {metrics['precision']:<10.2f} "
              f"{arrow} {delta_str}")
        prev_faith = metrics['faithfulness']
    
    print(f"\n{'='*60}")
    print("✅ RAG Evaluation Demo completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
