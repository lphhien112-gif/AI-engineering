# 🧱 00 — Foundations (Nền Tảng)

> Kiến thức nền tảng **bắt buộc** cho mọi AI Engineer.
> Master module này trước khi chuyển sang bất kỳ specialization nào.

---

## 🗺️ Learning Path

```
Bắt đầu ở đây!

Python Advanced ──→ Math for AI ──→ Git Workflow
      │                                   │
      ▼                                   ▼
  API Design ←── SQL Essentials ←── Docker
      │
      ▼
Data Engineering
      │
      ▼
  Interview Prep (foundations_qa.md)
```

**Thời gian ước tính**: 4-6 ngày (2-3 giờ/ngày)

---

## 📚 Docs

| # | File | Chủ đề | Lines | Thời gian |
|---|------|--------|:-----:|:---------:|
| 01 | [python_advanced.md](docs/01_python_advanced.md) | OOP, Metaclass, Async, Decorators, Testing, Logging | 570+ | ~25 min |
| 02 | [math_for_ai.md](docs/02_math_for_ai.md) | Linear Algebra, SVD, Probability, Information Theory | 480+ | ~25 min |
| 03 | [git_workflow.md](docs/03_git_workflow.md) | Git, Branching, Rebase, Hooks, LFS, Trunk-based | 350+ | ~15 min |
| 04 | [docker_essentials.md](docs/04_docker_essentials.md) | Dockerfile, Compose, GPU, Multi-stage, Health Checks | 400+ | ~20 min |
| 05 | [api_design.md](docs/05_api_design.md) | REST, FastAPI, CRUD, Auth, Middleware, Testing | 440+ | ~20 min |
| 06 | [sql_essentials.md](docs/06_sql_essentials.md) | JOINs, Window Functions, CTE, JSONB, Optimization | 420+ | ~20 min |
| 07 | [data_engineering.md](docs/07_data_engineering.md) | Spark, Kafka, ETL/ELT, Lakehouse, Airflow | 314 | ~15 min |

---

## 💻 Examples (Chạy được ngay!)

```bash
cd 00_foundations/examples

# Python — async patterns, semaphore, TaskGroup
python python_async_demo.py

# Math — vectors, SVD, PCA, softmax, information theory  
python math_demo.py           # cần: pip install numpy

# SQL — JOINs, window functions, CTEs, JSONB (dùng SQLite, zero deps!)
python sql_demo.py

# FastAPI — CRUD, auth, middleware
python fastapi_demo.py        # cần: pip install fastapi uvicorn
```

---

## 🎯 Interview Prep

| File | Số câu | Độ sâu |
|------|:------:|--------|
| [foundations_qa.md](interview/foundations_qa.md) | 45+ | Chi tiết — code examples, follow-ups, comparison tables |

**Topics**: Python (15), SQL (8), Docker (7), Git (5), Data Engineering (5), API Design (5)

---

## ✅ Checklist

### Phase 1: Đọc & Hiểu
- [ ] 01 - Python: OOP, Metaclass, Async, Decorators, Testing
- [ ] 02 - Math: Linear Algebra, Probability, Gradient, SVD  
- [ ] 03 - Git: Branching, Rebase, Conventional Commits
- [ ] 04 - Docker: Dockerfile, Compose, Multi-stage, GPU
- [ ] 05 - API: FastAPI, CRUD, Auth, Middleware, Testing
- [ ] 06 - SQL: JOINs, Window Functions, CTE, JSONB
- [ ] 07 - Data Engineering: Spark, Kafka, ETL/ELT

### Phase 2: Hands-on
- [ ] Chạy `math_demo.py` — hiểu SVD, softmax temperature
- [ ] Chạy `sql_demo.py` — thực hành window functions, CTE
- [ ] Chạy `python_async_demo.py` — hiểu asyncio.gather, semaphore
- [ ] Chạy `fastapi_demo.py` — test API endpoints

### Phase 3: Interview
- [ ] Trả lời 45+ câu hỏi trong `foundations_qa.md`
- [ ] Tự viết code cho 5 patterns: decorator factory, generator, context manager, async semaphore, pytest fixture
