"""
SQL Practice -- Self-contained SQLite Demo
Chay: python sql_demo.py

Demo: JOINs, Window Functions, CTEs, JSONB-like, Optimization
Tao in-memory DB, khong can install gi ngoai Python built-in.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import sqlite3
import json
from datetime import datetime, timedelta
import random

# ════════════════════════════════════════════
# SETUP: Create in-memory database
# ════════════════════════════════════════════

def create_database():
    """Create sample ML-themed database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript("""
        CREATE TABLE models (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            framework TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            model_id INTEGER REFERENCES models(id),
            input_text TEXT,
            prediction TEXT,
            confidence REAL,
            correct INTEGER,
            predicted_at TEXT
        );
        
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary REAL,
            manager_id INTEGER REFERENCES employees(id)
        );
        
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY,
            name TEXT,
            config TEXT,  -- JSON string (SQLite doesn't have JSONB)
            metrics TEXT, -- JSON string
            created_at TEXT
        );
    """)
    
    # Seed data
    models = [
        (1, "bert-base", "v1.0", "pytorch"),
        (2, "bert-large", "v2.0", "pytorch"),
        (3, "distilbert", "v1.0", "pytorch"),
        (4, "gpt-mini", "v1.0", "jax"),
    ]
    cursor.executemany("INSERT INTO models VALUES (?,?,?,?,datetime('now'))", models)
    
    # Generate predictions
    random.seed(42)
    predictions = []
    for i in range(1, 201):
        model_id = random.choice([1, 1, 1, 2, 2, 3])  # bert-base most used
        correct = random.random() > 0.2 if model_id <= 2 else random.random() > 0.35
        confidence = random.uniform(0.5, 0.99) if correct else random.uniform(0.3, 0.7)
        days_ago = random.randint(0, 30)
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        predictions.append((
            i, model_id, f"sample text {i}",
            "positive" if random.random() > 0.4 else "negative",
            round(confidence, 3), int(correct), date
        ))
    cursor.executemany(
        "INSERT INTO predictions VALUES (?,?,?,?,?,?,?)", predictions
    )
    
    # Employees (for window function demos)
    employees = [
        (1, "Alice", "Engineering", 120000, None),
        (2, "Bob", "Engineering", 100000, 1),
        (3, "Carol", "Engineering", 110000, 1),
        (4, "Dave", "Engineering", 95000, 1),
        (5, "Eve", "Data Science", 130000, None),
        (6, "Frank", "Data Science", 105000, 5),
        (7, "Grace", "Data Science", 115000, 5),
        (8, "Heidi", "Product", 90000, None),
        (9, "Ivan", "Product", 85000, 8),
    ]
    cursor.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", employees)
    
    # Experiments
    experiments = [
        (1, "bert-lr-1e-4", '{"model":"bert","lr":0.0001,"epochs":10}', '{"accuracy":0.89,"f1":0.85}', "2026-01-15"),
        (2, "bert-lr-5e-5", '{"model":"bert","lr":0.00005,"epochs":10}', '{"accuracy":0.92,"f1":0.90}', "2026-01-16"),
        (3, "distilbert-lr-2e-5", '{"model":"distilbert","lr":0.00002,"epochs":15}', '{"accuracy":0.87,"f1":0.83}', "2026-01-17"),
        (4, "bert-large-lr-3e-5", '{"model":"bert-large","lr":0.00003,"epochs":5}', '{"accuracy":0.94,"f1":0.92}', "2026-01-18"),
    ]
    cursor.executemany("INSERT INTO experiments VALUES (?,?,?,?,?)", experiments)
    
    conn.commit()
    return conn


def run_query(conn, title, sql, params=None):
    """Helper: run query and print results."""
    print(f"\n{'─' * 60}")
    print(f"📋 {title}")
    print(f"{'─' * 60}")
    print(f"  SQL: {sql.strip()[:120]}{'...' if len(sql.strip()) > 120 else ''}")
    
    cursor = conn.cursor()
    cursor.execute(sql, params or [])
    rows = cursor.fetchall()
    
    if not rows:
        print("  (no results)")
        return
    
    # Print header
    columns = [desc[0] for desc in cursor.description]
    widths = []
    for ci, col in enumerate(columns):
        col_vals = [len(str(row[ci])) for row in rows]
        max_w = max(len(col), max(col_vals) if col_vals else 0)
        widths.append(min(max_w + 2, 25))
    
    header = "  " + "".join(f"{col:<{w}}" for col, w in zip(columns, widths))
    print(header)
    print("  " + "".join("─" * w for w in widths))
    
    for row in rows[:15]:  # Limit output
        line = "  " + "".join(f"{str(row[i]):<{w}}" for i, w in enumerate(widths))
        print(line)
    
    if len(rows) > 15:
        print(f"  ... and {len(rows) - 15} more rows")
    print(f"  ({len(rows)} rows total)")


# ════════════════════════════════════════════
# DEMOS
# ════════════════════════════════════════════

def demo_joins(conn):
    """JOIN operations."""
    print("\n" + "=" * 60)
    print("1. JOINs")
    print("=" * 60)
    
    run_query(conn,
        "INNER JOIN: Models with predictions",
        """SELECT m.name, m.version, COUNT(p.id) AS prediction_count,
                  ROUND(AVG(p.confidence), 3) AS avg_confidence
           FROM models m
           INNER JOIN predictions p ON m.id = p.model_id
           GROUP BY m.id
           ORDER BY prediction_count DESC""")
    
    run_query(conn,
        "LEFT JOIN: ALL models (even without predictions)",
        """SELECT m.name, m.version, 
                  COALESCE(COUNT(p.id), 0) AS prediction_count
           FROM models m
           LEFT JOIN predictions p ON m.id = p.model_id
           GROUP BY m.id
           ORDER BY prediction_count DESC""")
    
    run_query(conn,
        "Self-JOIN: Employees earning more than their manager",
        """SELECT e.name AS employee, e.salary AS emp_salary,
                  m.name AS manager, m.salary AS mgr_salary,
                  e.salary - m.salary AS difference
           FROM employees e
           JOIN employees m ON e.manager_id = m.id
           WHERE e.salary > m.salary""")


def demo_window_functions(conn):
    """Window functions: RANK, LAG, running totals."""
    print("\n" + "=" * 60)
    print("2. WINDOW FUNCTIONS")
    print("=" * 60)
    
    run_query(conn,
        "RANK employees by salary within department",
        """SELECT name, department, salary,
                  RANK() OVER(PARTITION BY department ORDER BY salary DESC) AS rank,
                  DENSE_RANK() OVER(PARTITION BY department ORDER BY salary DESC) AS dense_rank
           FROM employees
           ORDER BY department, rank""")
    
    run_query(conn,
        "Model accuracy over time (daily aggregate with LAG)",
        """SELECT 
               predicted_at AS date,
               COUNT(*) AS total,
               ROUND(AVG(correct) * 100, 1) AS accuracy_pct,
               LAG(ROUND(AVG(correct) * 100, 1)) OVER(ORDER BY predicted_at) AS prev_day,
               ROUND(
                   AVG(correct) * 100 - 
                   LAG(AVG(correct) * 100) OVER(ORDER BY predicted_at), 1
               ) AS change
           FROM predictions
           WHERE model_id = 1
           GROUP BY predicted_at
           ORDER BY date
           LIMIT 10""")
    
    run_query(conn,
        "Running total of predictions per model",
        """SELECT predicted_at, model_id, COUNT(*) AS daily_count,
                  SUM(COUNT(*)) OVER(
                      PARTITION BY model_id 
                      ORDER BY predicted_at 
                      ROWS UNBOUNDED PRECEDING
                  ) AS running_total
           FROM predictions
           WHERE model_id IN (1, 2)
           GROUP BY predicted_at, model_id
           ORDER BY model_id, predicted_at
           LIMIT 12""")


def demo_ctes(conn):
    """CTEs and subqueries."""
    print("\n" + "=" * 60)
    print("3. CTEs (Common Table Expressions)")
    print("=" * 60)
    
    run_query(conn,
        "CTE: Model performance summary with ranking",
        """WITH model_stats AS (
               SELECT model_id,
                      COUNT(*) AS total,
                      SUM(correct) AS correct_count,
                      ROUND(AVG(correct) * 100, 1) AS accuracy,
                      ROUND(AVG(confidence), 3) AS avg_confidence
               FROM predictions
               GROUP BY model_id
           )
           SELECT m.name, m.version,
                  ms.total, ms.accuracy,
                  ms.avg_confidence,
                  RANK() OVER(ORDER BY ms.accuracy DESC) AS perf_rank
           FROM model_stats ms
           JOIN models m ON m.id = ms.model_id
           ORDER BY perf_rank""")
    
    run_query(conn,
        "Recursive CTE: Organization tree",
        """WITH RECURSIVE org_tree AS (
               SELECT id, name, manager_id, 0 AS level, name AS path
               FROM employees
               WHERE manager_id IS NULL
               
               UNION ALL
               
               SELECT e.id, e.name, e.manager_id, t.level + 1,
                      t.path || ' → ' || e.name
               FROM employees e
               JOIN org_tree t ON e.manager_id = t.id
           )
           SELECT name, level, 
                  SUBSTR('          ', 1, level * 2) || name AS indented,
                  path
           FROM org_tree
           ORDER BY path""")


def demo_aggregation(conn):
    """GROUP BY, HAVING, aggregation functions."""
    print("\n" + "=" * 60)
    print("4. AGGREGATION & FILTERING")
    print("=" * 60)
    
    run_query(conn,
        "HAVING: Only models with accuracy > 75%",
        """SELECT m.name,
                  COUNT(p.id) AS total,
                  ROUND(AVG(p.correct) * 100, 1) AS accuracy
           FROM models m
           JOIN predictions p ON m.id = p.model_id
           GROUP BY m.id
           HAVING AVG(p.correct) > 0.75
           ORDER BY accuracy DESC""")
    
    run_query(conn,
        "High-confidence wrong predictions (model debugging)",
        """SELECT m.name AS model, p.input_text, 
                  p.prediction, p.confidence,
                  CASE WHEN p.correct THEN '✅' ELSE '❌' END AS result
           FROM predictions p
           JOIN models m ON m.id = p.model_id
           WHERE p.correct = 0 AND p.confidence > 0.7
           ORDER BY p.confidence DESC
           LIMIT 10""")


def demo_json(conn):
    """JSON queries (simulated JSONB)."""
    print("\n" + "=" * 60)
    print("5. JSON QUERIES (PostgreSQL JSONB equivalent)")
    print("=" * 60)
    
    run_query(conn,
        "Extract JSON fields from experiment configs",
        """SELECT name,
                  json_extract(config, '$.model') AS model,
                  json_extract(config, '$.lr') AS learning_rate,
                  json_extract(config, '$.epochs') AS epochs,
                  json_extract(metrics, '$.accuracy') AS accuracy,
                  json_extract(metrics, '$.f1') AS f1_score
           FROM experiments
           ORDER BY json_extract(metrics, '$.accuracy') DESC""")


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

if __name__ == "__main__":
    print("🗃️ SQL PRACTICE — Self-Contained SQLite Demo")
    print("=" * 60)
    print("Creating in-memory database with ML-themed data...")
    
    conn = create_database()
    
    # Verify data
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    pred_count = cursor.fetchone()[0]
    print(f"  ✅ {pred_count} predictions loaded")
    print(f"  ✅ 4 models, 9 employees, 4 experiments")
    
    demo_joins(conn)
    demo_window_functions(conn)
    demo_ctes(conn)
    demo_aggregation(conn)
    demo_json(conn)
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ All SQL demos completed!")
    print("Key takeaways:")
    print("  • JOIN: link related tables (INNER vs LEFT)")
    print("  • Window Functions: aggregate WITHOUT reducing rows")
    print("  • CTE: readable named subqueries (recursive possible)")
    print("  • HAVING: filter AFTER GROUP BY")
    print("  • JSON: flexible schema for ML configs/metrics")
    print("  • Practice these patterns for interviews!")
    print("=" * 60)
