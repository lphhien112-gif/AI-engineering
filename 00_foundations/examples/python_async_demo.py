"""
🐍 Python Advanced Demo — Async, Decorators, Generators, Type Hints
Chạy: python python_async_demo.py
"""
import asyncio
import functools
import time
import sys
from typing import Optional, Callable, Protocol, Any
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field


# ============================================================
# 1. DECORATORS
# ============================================================

def timer(func):
    """Decorator đo thời gian chạy."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  ⏱️  {func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 0.1):
    """Decorator factory — retry khi gặp exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"  ⚠️  Attempt {attempt}/{max_attempts}: {e}")
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator


@timer
def slow_function():
    """Demo timer decorator."""
    time.sleep(0.3)
    return "done"


@retry(max_attempts=3, delay=0.1)
def flaky_function():
    """Demo retry decorator."""
    import random
    if random.random() < 0.6:
        raise ConnectionError("Random failure")
    return "success!"


# ============================================================
# 2. ASYNC / AWAIT
# ============================================================

async def fetch_data(name: str, delay: float) -> dict:
    """Simulate async API call."""
    print(f"  🔄 Fetching {name}...")
    await asyncio.sleep(delay)  # Non-blocking sleep
    return {"name": name, "status": "ok"}


async def async_demo():
    """Demo asyncio.gather — chạy đồng thời."""
    print("\n--- Async Demo ---")
    
    # Sequential (slow)
    start = time.perf_counter()
    r1 = await fetch_data("model_A", 0.3)
    r2 = await fetch_data("model_B", 0.3)
    r3 = await fetch_data("model_C", 0.3)
    seq_time = time.perf_counter() - start
    print(f"  Sequential: {seq_time:.2f}s")
    
    # Concurrent (fast)
    start = time.perf_counter()
    results = await asyncio.gather(
        fetch_data("model_A", 0.3),
        fetch_data("model_B", 0.3),
        fetch_data("model_C", 0.3),
    )
    conc_time = time.perf_counter() - start
    print(f"  Concurrent: {conc_time:.2f}s")
    print(f"  Speedup: {seq_time / conc_time:.1f}x")
    return results


# ============================================================
# 3. GENERATORS
# ============================================================

def fibonacci(limit: int):
    """Generator — lazy Fibonacci sequence."""
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b


def batch_data(data: list, batch_size: int = 3):
    """Generator — chia data thành batches."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def generator_demo():
    """Demo generators vs list comprehension memory."""
    print("\n--- Generator Demo ---")
    
    # Fibonacci
    fibs = list(fibonacci(100))
    print(f"  Fibonacci < 100: {fibs}")
    
    # Batching
    data = list(range(10))
    print(f"  Data: {data}")
    for i, batch in enumerate(batch_data(data, 3)):
        print(f"    Batch {i}: {batch}")
    
    # Memory comparison
    n = 1_000_000
    list_size = sys.getsizeof([x**2 for x in range(n)])
    gen_size = sys.getsizeof(x**2 for x in range(n))
    print(f"  List ({n} items): {list_size:,} bytes")
    print(f"  Generator: {gen_size:,} bytes")
    print(f"  Memory saved: {(1 - gen_size/list_size)*100:.1f}%")


# ============================================================
# 4. TYPE HINTS + DATACLASSES
# ============================================================

class Predictable(Protocol):
    """Protocol — structural subtyping (duck typing + type checking)."""
    def predict(self, X: list) -> list: ...


@dataclass
class ModelConfig:
    """Dataclass — auto __init__, __repr__, __eq__."""
    name: str
    learning_rate: float = 0.001
    epochs: int = 10
    tags: list[str] = field(default_factory=list)


class SimpleModel:
    """A simple model that satisfies the Predictable protocol."""
    def predict(self, X: list) -> list:
        return [x * 2 for x in X]


def evaluate(model: Predictable, data: list) -> list:
    """Accepts any object with a predict() method."""
    return model.predict(data)


def type_hints_demo():
    """Demo type hints + dataclasses."""
    print("\n--- Type Hints & Dataclass Demo ---")
    
    config = ModelConfig("transformer", learning_rate=0.0001, tags=["nlp", "prod"])
    print(f"  Config: {config}")
    
    model = SimpleModel()
    result = evaluate(model, [1, 2, 3, 4, 5])
    print(f"  Predictions: {result}")


# ============================================================
# 5. STANDARD LIBRARY 
# ============================================================

def stdlib_demo():
    """Demo useful stdlib tools."""
    print("\n--- Standard Library Demo ---")
    
    # Counter
    tokens = ["the", "model", "is", "the", "best", "model", "for", "the", "task"]
    freq = Counter(tokens)
    print(f"  Top 3 tokens: {freq.most_common(3)}")
    
    # defaultdict
    doc_embeddings = defaultdict(list)
    doc_embeddings["doc_1"].extend([0.1, 0.2, 0.3])
    doc_embeddings["doc_2"].extend([0.4, 0.5, 0.6])
    print(f"  Embeddings: dict({dict(doc_embeddings)})")
    
    # deque (sliding window)
    window = deque(maxlen=5)
    for i in range(8):
        window.append(i)
    print(f"  Sliding window (maxlen=5): {list(window)}")
    
    # lru_cache
    @functools.lru_cache(maxsize=32)
    def compute_embedding(text: str) -> int:
        """Cached computation."""
        time.sleep(0.1)  # Simulate expensive call
        return hash(text) % 1000
    
    start = time.perf_counter()
    compute_embedding("hello world")
    first_call = time.perf_counter() - start
    
    start = time.perf_counter()
    compute_embedding("hello world")  # Cached!
    second_call = time.perf_counter() - start
    
    print(f"  lru_cache: 1st call {first_call:.4f}s, 2nd call {second_call:.6f}s (cached)")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🐍 Python Advanced Demo")
    print("=" * 60)
    
    # 1. Decorators
    print("\n--- Decorator Demo ---")
    slow_function()
    try:
        flaky_function()
        print("  ✅ flaky_function succeeded")
    except ConnectionError:
        print("  ❌ flaky_function failed after all retries")
    
    # 2. Async
    asyncio.run(async_demo())
    
    # 3. Generators
    generator_demo()
    
    # 4. Type hints
    type_hints_demo()
    
    # 5. Stdlib
    stdlib_demo()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60)
