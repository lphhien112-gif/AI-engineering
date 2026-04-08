# 🎯 Foundations — Câu Hỏi Phỏng Vấn (50+)

> Mỗi câu hỏi có gợi ý trả lời **CHI TIẾT**, follow-up questions, và ví dụ thực tế.
> Khi phỏng vấn: giải thích → ví dụ → trade-off → follow-up.

---

## Python (15 câu)

### Q1: `__new__` vs `__init__` khác nhau thế nào?
**A**: 
- `__new__(cls)`: tạo instance (allocate memory), được gọi TRƯỚC, return instance
- `__init__(self)`: khởi tạo instance đã tạo, KHÔNG return gì
- `__new__` hiếm dùng: Singleton pattern, subclass immutable types (str, tuple, int)
```python
class Singleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance  # Luôn trả cùng instance
```
- **Follow-up**: "Metaclass dùng `__new__` thế nào?" → Metaclass `__new__` tạo CLASS object (không phải instance). Dùng để modify class creation (add methods, validate, register).

### Q2: GIL là gì? Tại sao Python có GIL?
**A**: Global Interpreter Lock — mutex cho phép chỉ 1 thread thực thi Python bytecode tại 1 thời điểm.
- **Tại sao**: Reference counting (Python memory management) không thread-safe → GIL bảo vệ
- **Impact**: CPU-bound tasks KHÔNG benefit từ threading
- **Bypass**: (1) `multiprocessing` (separate processes, separate GIL), (2) C extensions (NumPy, PyTorch release GIL), (3) async I/O (not CPU-bound)
- **Python 3.13+**: PEP 703 — Free-threaded Python (experimental, disable GIL)
- **Follow-up**: "Tại sao threading vẫn useful dù có GIL?" → I/O operations RELEASE GIL. API calls, file I/O, DB queries → threads chạy song song thực sự.

### Q3: `asyncio` vs `threading` vs `multiprocessing`?
**A**:
| | asyncio | threading | multiprocessing |
|-|---------|-----------|-----------------|
| **Best for** | I/O-bound (APIs, DB) | Simple I/O | CPU-bound (training) |
| **Concurrency** | Cooperative (await) | Preemptive (OS) | True parallelism |
| **GIL** | Not relevant | Blocked | Bypassed |
| **Overhead** | Lowest (no OS threads) | Medium | High (process spawn) |
| **Sharing** | Same memory | Same memory (careful!) | Separate memory (IPC) |
- **Rule of thumb**: API calls → async. File I/O → threading. ML training → multiprocessing.
- **Follow-up**: "uvloop vs asyncio?" → uvloop = C implementation, 2-3x faster. Auto-used by uvicorn.

### Q4: Decorator factory — viết retry decorator
**A**:
```python
import functools, time

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator FACTORY — returns a decorator."""
    def decorator(func):
        @functools.wraps(func)  # BẮT BUỘC: preserve func metadata
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        raise
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def call_api():
    ...
```
- 3 levels: `retry()` → `decorator()` → `wrapper()`
- **Follow-up**: "Async decorator?" → Same structure nhưng `async def wrapper` và `await func()`

### Q5: Generator — tại sao quan trọng cho ML?
**A**: Lazy evaluation — tính từng item khi cần, ~0 memory.
```python
# ❌ Load 10GB dataset vào RAM
all_data = [process(line) for line in open("huge.csv")]  # 10GB in memory!

# ✅ Generator — process 1 line at a time
def data_generator(filepath):
    with open(filepath) as f:
        for line in f:
            yield process(line)  # ~0 memory per iteration

# PyTorch DataLoader DÙNG generators internally
for batch in DataLoader(dataset, batch_size=32):
    train(batch)  # Only 32 samples in memory
```
- **Follow-up**: "yield vs yield from?" → `yield from iterable` = delegate, flatten nested generators.

### Q6-Q15 (Chi tiết hơn)

### Q6: `yield from` dùng khi nào?
**A**: Delegate cho sub-generator, flatten nested structures.
```python
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)  # Delegate to recursive call
        else:
            yield item

list(flatten([[1, [2, 3]], [4, [5, [6]]]])) # [1, 2, 3, 4, 5, 6]
```

### Q7: Protocol (structural subtyping)?
**A**: "Duck typing with type safety". Object chỉ cần implement methods, không cần inherit.
```python
from typing import Protocol

class Predictable(Protocol):
    def predict(self, X) -> list: ...

def evaluate(model: Predictable):  # Accepts ANY object with predict()
    return model.predict(test_data)

# Cả sklearn model VÀ custom model đều work — no inheritance needed!
```
- Khác ABC: ABC requires `class MyModel(ABC)`. Protocol requires nothing — just implement methods.

### Q8: Pydantic vs dataclass?
**A**: 
- **Pydantic**: RUNTIME validation, type coercion (`"123"` → `123`), custom validators, JSON schema. V2 dùng Rust → 5-50x faster.
- **dataclass**: Code generation (`__init__`, `__repr__`), NO validation, lighter, stdlib.
- **Rule**: API input/output → Pydantic. Internal configs, data containers → dataclass.
- **Follow-up**: "attrs vs dataclass?" → attrs = more features (validators, converters, slots by default).

### Q9: `lru_cache` hoạt động thế nào?
**A**: Least Recently Used cache — memoize function results.
```python
from functools import lru_cache

@lru_cache(maxsize=128)  # Cache 128 most recent results
def expensive_computation(n: int) -> int:
    return n ** 2  # Only computed once per unique n

# ⚠️ Arguments MUST be hashable (no lists, dicts)
# ⚠️ Don't cache functions with side effects
# ⚠️ memory grows — use maxsize to limit
```

### Q10: Context Manager — custom example?
**A**: Quản lý resources — đảm bảo cleanup even if exception.
```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    """Đo thời gian execution."""
    start = time.perf_counter()
    yield  # Code block runs here
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f}s")

# Usage
with timer("Training"):
    model.fit(X, y)

# Class-based:
class DBConnection:
    def __enter__(self):
        self.conn = connect()
        return self.conn
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()  # ALWAYS cleanup, even if exception
        return False  # Don't suppress exceptions
```

### Q11-Q15

**Q11: `*args`, `**kwargs`?** → `*args` = tuple of positional args. `**kwargs` = dict of keyword args. Essential for: decorators, function wrappers, flexible APIs. Order: `func(pos, *args, key=val, **kwargs)`.

**Q12: Shallow vs Deep copy?** → Shallow: new container, same nested objects (references shared). Deep: new everything (independent). `import copy; copy.copy()` vs `copy.deepcopy()`. Pitfall: list slicing `a[:]` = shallow copy.

**Q13: List vs Tuple vs Set?** → List: mutable, ordered, O(1) append, O(n) search. Tuple: immutable, hashable (dict key), namedtuple for structured data. Set: unique, O(1) lookup/add, unordered. frozenset = immutable set.

**Q14: `__str__` vs `__repr__`?** → `str()`: user-friendly. `repr()`: developer, unambiguous, ideally `eval(repr(obj)) == obj`. Rule: always implement `__repr__`. `__str__` falls back to `__repr__`.

**Q15: Walrus operator `:=`?** → Assign + use in same expression: `while chunk := f.read(8192):`. In comprehensions: `[y for x in data if (y := transform(x)) > 0]`. Avoid overuse — readability first.

---

## SQL (8 câu)

### Q16: INNER JOIN vs LEFT JOIN?
**A**: 
- **INNER**: ONLY rows with matches in BOTH tables. Use: "show me orders WITH customers"
- **LEFT**: ALL rows from left table + matching from right (NULL if no match). Use: "show ALL customers, even without orders"
- **RIGHT**: Mirror of LEFT (rarely used, rearrange table order instead)
- **FULL OUTER**: Everything from both (NULLs on both sides if no match)
- **Follow-up**: "Performance difference?" → INNER usually faster (fewer rows to process). LEFT JOIN on indexed columns ≈ INNER speed.

### Q17: Window Functions — ví dụ ML?
**A**: Aggregate WITHOUT reducing rows. Pattern: `function() OVER(PARTITION BY col ORDER BY col)`.
```sql
-- Model performance ranking per day
SELECT model_name, date, accuracy,
       RANK() OVER(PARTITION BY date ORDER BY accuracy DESC) AS daily_rank,
       LAG(accuracy) OVER(PARTITION BY model_name ORDER BY date) AS prev_accuracy,
       accuracy - LAG(accuracy) OVER(PARTITION BY model_name ORDER BY date) AS improvement
FROM model_metrics;
```
- Key functions: ROW_NUMBER, RANK, DENSE_RANK, LAG/LEAD, NTILE, SUM/AVG/COUNT() OVER

### Q18: CTE vs Subquery?
**A**: 
- **CTE** (`WITH ... AS`): named, readable, reusable within query, supports RECURSIVE
- **Subquery**: inline, sometimes marginally faster, no recursion
- **PostgreSQL**: CTEs were materialized (slower) before v12. Now optimized.
- **Rule**: CTE cho complex multi-step queries. Subquery cho simple filters.

### Q19: Index — khi nào nên và không nên?
**A**: B-tree index = sorted tree structure, O(log n) lookup.
- **Nên index**: WHERE columns, JOIN keys, ORDER BY columns, high cardinality (unique values)
- **KHÔNG index**: (1) Boolean/gender (low cardinality), (2) Tables <1000 rows, (3) Frequently updated columns (index maintenance cost), (4) Quá nhiều indexes (slow writes)
- **Composite index** `(a, b, c)`: works for WHERE a=?, WHERE a=? AND b=?, nhưng KHÔNG cho WHERE b=? alone (leftmost prefix rule)

### Q20-Q23

**Q20: HAVING vs WHERE?** → WHERE: filter TRƯỚC GROUP BY (trên rows). HAVING: filter SAU GROUP BY (trên aggregated results). Example: `HAVING COUNT(*) > 5` (filter groups with >5 members).

**Q21: EXPLAIN ANALYZE?** → Shows execution plan: Seq Scan (bad) vs Index Scan (good), estimated vs actual rows, cost. Use to optimize: add indexes, rewrite queries, check join types.

**Q22: N+1 Query Problem?** → 1 query for N items + N queries for details = N+1 total. Fix: JOINs in SQL, or `select_related()`/`prefetch_related()` in Django ORM. Very common in ORMs.

**Q23: JSONB (PostgreSQL)?** → Semi-structured data: ML configs, metadata. GIN index for fast `@>` queries. `config->>'model'` extracts text. Don't use for structured, queryable data — proper columns perform better.

---

## Docker (7 câu)

### Q24: Image vs Container?
**A**: Image = read-only blueprint (multiple layers, shareable, versioned). Container = running instance (writable layer on top, ephemeral). Analogy: Class vs Object, Recipe vs Dish.
- **Follow-up**: "Layers?" → Each Dockerfile instruction = 1 layer. Layers stack and cache independently. COPY requirements + pip install = 2 cached layers when source changes.

### Q25-Q30

**Q25: Multi-stage build?** → Stage 1: install ALL deps, compile, test. Stage 2: copy ONLY runtime artifacts. Result: 60-70% smaller, no build tools/pip in production → smaller attack surface.

**Q26: COPY vs ADD?** → COPY: explicit file copy. ADD: copy + auto-extract tar + download URL. **Best practice: always COPY**. ADD only when extracting tar needed.

**Q27: CMD vs ENTRYPOINT?** → ENTRYPOINT: fixed command (executable). CMD: default args (overridable). Pattern: `ENTRYPOINT ["python"]` + `CMD ["app.py"]` → user can `docker run image test.py` (overrides CMD).

**Q28: Docker Compose?** → Multi-container orchestration in YAML. Services + networks + volumes. `docker compose up -d` starts everything. Good for: API + DB + Redis + VectorDB stacks.

**Q29: Volume vs Bind Mount?** → Volume: Docker-managed, persists, best for production data (DB). Bind mount: maps host dir → container, best for development (live code reload). Named volumes survive container removal.

**Q30: Reduce image size?** → (1) Multi-stage builds, (2) `python:3.11-slim` not `python:3.11`, (3) `--no-cache-dir` for pip, (4) `.dockerignore` (exclude tests, docs, data), (5) Combine RUN commands (fewer layers), (6) Clean apt cache.

---

## Git (5 câu)

### Q31: Merge vs Rebase?
**A**: 
- **Merge**: tạo merge commit, giữ nguyên lịch sử branches → safe cho shared branches
- **Rebase**: rewrite commits, tạo linear history → clean nhưng DESTRUCTIVE
- **GOLDEN RULE**: Không bao giờ rebase branch đã push mà người khác đang dùng!
- **Workflow**: rebase local branch trước khi push → clean PR → merge to main
- **Follow-up**: "Interactive rebase?" → `git rebase -i HEAD~3` — squash, reword, reorder, drop commits

### Q32-Q35

**Q32: Conventional Commits?** → `type(scope): description`. Benefits: auto changelog, semantic versioning (feat=minor, fix=patch, BREAKING=major), focused commits. ML-specific types: experiment, model, data.

**Q33: Cherry-pick?** → `git cherry-pick abc123` — apply 1 specific commit từ branch khác. Use case: hotfix from develop to main, bring specific feature without merging entire branch.

**Q34: Stash?** → `git stash push -m "WIP"` — save uncommitted changes temporarily. `git stash pop` — restore. Use: need to switch branch urgently without committing half-done work.

**Q35: .gitignore ML?** → `*.pth *.onnx *.safetensors` (models), `data/ checkpoints/` (large files), `wandb/ mlruns/` (experiment tracking), `.env` (secrets), `__pycache__/` (Python).

---

## Data Engineering (5 câu)

### Q36: ETL vs ELT?
**A**: 
- **ETL**: Extract → Transform → Load. Transform TRƯỚC load. Tool: Spark, Airflow. Legacy, on-prem.
- **ELT**: Extract → Load → Transform. Load raw data → transform IN warehouse. Tool: dbt, BigQuery. Modern, cloud.
- **AI Engineer nên biết ELT**: (1) Giữ raw data cho future retraining, (2) Transform logic versioned (dbt), (3) Schema evolution flexible.

### Q37-Q40

**Q37: Spark?** → Distributed processing. Driver → Executors → Tasks. Lazy evaluation (build DAG → execute on action). DataFrame API ≈ pandas nhưng distributed trên cluster. For AI: feature engineering at scale, data preprocessing.

**Q38: Kafka?** → Distributed event streaming. Producer → Topic (partitions) → Consumer Groups. Use cases: real-time feature serving, prediction logging, event-driven ML pipelines. Guarantees: at-least-once, exactly-once (with config).

**Q39: Data Lakehouse?** → Lake (raw, cheap, S3/GCS) + Warehouse (structured, ACID, query). Formats: Delta Lake (Databricks), Apache Iceberg. Features: time travel (ml reproducibility), schema evolution, ACID on files.

**Q40: Data quality?** → "Garbage In, Garbage Out". Validation layers: (1) Schema validation (correct types), (2) Distribution checks (drift), (3) Freshness (is data recent?), (4) Completeness (missing values). Tools: Great Expectations, dbt tests.

---

## API Design (5 câu)

### Q41: REST best practices?
**A**: 
- Noun-based URLs: `/api/v1/models/123` (not `/getModel`)
- HTTP methods: GET (read), POST (create), PUT (full update), PATCH (partial), DELETE (remove)
- Status codes: 200 OK, 201 Created, 400 Bad Request, 401 Unauthorized, 404 Not Found, 500 Internal
- Versioning: `/api/v1/` (URL-based, most common)
- Pagination: `?page=2&limit=20` or cursor-based

### Q42: JWT flow?
**A**: Login → server creates JWT (header.payload.signature, signed with secret key) → client stores token → sends in `Authorization: Bearer <token>` header → server verifies signature (stateless, no DB lookup). Expiry: 15-30 min access token + longer refresh token.

### Q43: Rate limiting?
**A**: Protect API from abuse. Strategies: (1) Fixed window (100 req/min), (2) Sliding window (smoother), (3) Token bucket (burst-friendly). Implementation: Redis counters. Layers: Nginx/CDN (volumetric) + application (per-user, per-API-key).

### Q44: CORS là gì?
**A**: Cross-Origin Resource Sharing. Browser blocks requests from different domain by default (security). API server must explicitly allow origins via `Access-Control-Allow-Origin` header. Needed when: React app (localhost:3000) calls API (localhost:8000).

### Q45: Idempotent methods?
**A**: Calling multiple times = same result. GET, PUT, DELETE = idempotent. POST = NOT idempotent (mỗi call tạo resource mới). Importance: safe to retry failed requests for idempotent methods.

---

## Advanced Python — Bonus (7 câu)

### Q46: Metaclass vs `__init_subclass__`?
**A**: 
- **Metaclass**: full control over class creation — modify `__new__`, `__init__` of the CLASS itself. Used by Django ORM, SQLAlchemy. Complex, "dark magic".
- **`__init_subclass__`**: simpler hook — called when a subclass is created. Good for: auto-registration, validation, adding methods.
- **Rule**: `__init_subclass__` first. Metaclass only when you NEED to modify class creation mechanics.
- **Follow-up**: Real-world sử dụng? → Plugin systems, model registries, API endpoint auto-discovery.

### Q47: Descriptor protocol?
**A**: Objects that define `__get__`, `__set__`, `__delete__` control attribute access on OTHER classes.
- **Data descriptor**: has `__set__` or `__delete__` → takes priority over instance `__dict__`
- **Non-data descriptor**: only `__get__` → instance `__dict__` takes priority
- Powers: `@property`, `@classmethod`, `@staticmethod`, `__slots__`, Django model fields
- **Follow-up**: Lookup order? → Data descriptor → instance `__dict__` → non-data descriptor → `__getattr__`

### Q48: `__call__` trong ML?
**A**: Makes instances callable like functions. Pattern used everywhere in ML:
- `model(x)` → PyTorch `nn.Module.__call__` → `forward(x)` + hooks
- `preprocess(image)` → callable preprocessor with state
- `loss_fn(pred, target)` → stateful loss function
- Benefits: function-like API nhưng giữ state (config, params, history).

### Q49: pytest — fixture scopes?
**A**: `function` (default): run per test. `class`: per test class. `module`: per file. `session`: once for entire test run.
- Use `session` for: DB connection, model loading (expensive setup)
- Use `function` for: clean state (data, mocks)
- `conftest.py`: auto-shared fixtures across test files
- `yield` in fixture: setup → yield → teardown (cleanup)

### Q50: Structured logging vs print?
**A**: 
- `print()`: no timestamp, no level, no structure → useless in production
- `logging`: levels (DEBUG/INFO/WARNING/ERROR), timestamps, configurable output
- **structlog** (production): JSON output → parseable by ELK/Datadog. Key-value pairs. Correlation IDs.
- Rule: development → logging. Production → structlog with JSON.

### Q51: Exception chaining (`from`)?
**A**: `raise NewError("msg") from original_error` — chains exceptions, preserving traceback.
- Shows "The above exception was the direct cause of..."
- Useful: convert low-level exceptions to domain-specific ones
- `raise X from None` → suppress chaining (hide internal details from users)

### Q52: Walrus operator `:=` patterns?
**A**: Assign + use in same expression. Key patterns:
```python
# While loop
while chunk := f.read(8192):
    process(chunk)

# List comprehension with filter
results = [y for x in data if (y := expensive(x)) > threshold]

# Regex
if m := re.match(pattern, text):
    return m.group(1)
```
Rule: improves readability when value needed in BOTH condition AND body. Don't overuse.
