# 🖼️ Multi-modal RAG — Production Guide

> **Mục tiêu**: RAG beyond text — Images, Tables, PDFs, Audio, Vision-Language models.
> Real-world documents = text + tables + images + charts. Multi-modal RAG handles ALL of them.

---

## 1. Multi-modal RAG Architecture

```mermaid
graph TB
    D["Documents<br/>PDF, DOCX, Images, Audio"] --> P["Intelligent Parsing"]
    
    P --> T["Text → Chunks → Embeddings"]
    P --> TB["Tables → NL Description → Embeddings"]
    P --> I["Images → Vision Model → Embeddings"]
    P --> A["Audio → Whisper → Embeddings"]
    
    T --> VS["Unified Vector Store<br/>pgvector / Qdrant"]
    TB --> VS
    I --> VS
    A --> VS
    
    VS --> RET["Retrieval<br/>text + images + tables"]
    RET --> LLM["Multi-modal LLM Context"]
    LLM --> ANS["Answer + citations"]
```

---

## 2. PDF Parsing — Deep Dive

### 2.1 Tool Comparison

| Tool | Speed | Quality | Tables | Images | OCR | Best For |
|------|:-----:|:-------:|:------:|:------:|:---:|---------|
| **PyMuPDF (fitz)** | ⚡⚡⚡ | Good | ✅ | ✅ | ❌ | Digital PDFs (text-based) |
| **Unstructured** | ⚡ | Best | ✅ | ✅ | ✅ | Complex layouts, mixed content |
| **LlamaParse** | ⚡ | Best | ✅ | ✅ | ✅ | Cloud API, easiest |
| **pdfplumber** | ⚡⚡ | Good | ✅⚡ | ❌ | ❌ | Table-heavy PDFs |
| **Docling (IBM)** | ⚡ | Best | ✅ | ✅ | ✅ | Academic papers, open-source |

### 2.2 PyMuPDF (Fast, Digital PDFs)

```python
import fitz  # PyMuPDF

def extract_pdf(path: str) -> list[dict]:
    doc = fitz.open(path)
    pages = []
    
    for page_num, page in enumerate(doc):
        # Text with layout preservation
        text = page.get_text("text")
        
        # Tables (built-in since PyMuPDF 1.23+)
        tables = page.find_tables()
        table_data = [table.extract() for table in tables]
        
        # Images
        images = page.get_images()
        image_list = []
        for img_idx, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            image_list.append({
                "index": img_idx,
                "width": pix.width,
                "height": pix.height,
                "bytes": pix.tobytes("png"),  # Raw image bytes
            })
        
        pages.append({
            "page": page_num + 1,
            "text": text,
            "tables": table_data,
            "images": image_list,
        })
    
    return pages
```

### 2.3 Unstructured (Complex Layouts)

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    "document.pdf",
    strategy="hi_res",              # OCR + layout analysis (best quality)
    infer_table_structure=True,     # Extract tables as HTML
    extract_images_in_pdf=True,     # Extract images
    include_page_breaks=True,
    languages=["eng", "vie"],       # Multi-language OCR
)

# Group by type
for element in elements:
    if element.category == "NarrativeText":
        print(f"[Text p{element.metadata.page_number}]: {str(element)[:100]}...")
    elif element.category == "Table":
        print(f"[Table p{element.metadata.page_number}]: {element.metadata.text_as_html[:100]}...")
    elif element.category == "Image":
        print(f"[Image p{element.metadata.page_number}]: {str(element)[:100]}...")
```

### 2.4 Docling (SOTA Open-Source, IBM)

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("paper.pdf")

# Export as markdown (preserves structure + tables)
markdown = result.document.export_to_markdown()

# Export as structured elements
for item in result.document.iterate_items():
    print(f"{item.label}: {item.text[:80]}...")
    if item.label == "table":
        # Access table as DataFrame
        df = item.export_to_dataframe()
        print(df.head())
```

---

## 3. Vision-Language RAG

### 3.1 Image Description Pipeline

```python
from openai import OpenAI
import base64

client = OpenAI()

def describe_image_for_rag(image_path: str) -> str:
    """Generate searchable, detailed description for an image."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": 
                    "Describe this image in detail for search indexing. Include: "
                    "1. What the image shows (objects, people, scenes) "
                    "2. Any text visible in the image "
                    "3. Data points if it's a chart/graph "
                    "4. Technical details if it's a diagram "
                    "Keep it factual and searchable."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        max_tokens=500,
    )
    return response.choices[0].message.content

# Pipeline:
# 1. Extract images from PDF
# 2. Describe each image with vision model  
# 3. Embed description text
# 4. Store: {description_embedding, original_image_bytes, page_number}
# 5. Retrieve by description similarity
# 6. Pass original image + description to multi-modal LLM
```

### 3.2 CLIP Embeddings (Cross-modal Search)

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# CLIP: embed text AND images in same space!
model = SentenceTransformer("clip-ViT-B-32")

# Embed text
text_emb = model.encode("a photo of a cat")

# Embed image
img = Image.open("cat.jpg")
img_emb = model.encode(img)

# Cross-modal search: text query → find similar images
# cosine_similarity(text_emb, img_emb) → high if matching

# Use cases:
# 1. "find graphs showing revenue growth" → retrieves chart images
# 2. "architecture diagram" → retrieves system diagrams
# 3. Combine with text embeddings for hybrid multimodal search
```

---

## 4. Table RAG — First-class Tables

```python
class TableRAG:
    """Handle tables as first-class citizens in RAG."""
    
    def process_table(self, table_data: list[list[str]], caption: str = "") -> dict:
        """Process a table into multiple representations."""
        
        # Strategy 1: Natural language (for embedding + retrieval)
        nl_description = self._table_to_nl(table_data, caption)
        
        # Strategy 2: Markdown (for LLM context)
        markdown = self._table_to_markdown(table_data, caption)
        
        # Strategy 3: Structured (for SQL-like queries)
        structured = {
            "headers": table_data[0],
            "rows": table_data[1:],
            "caption": caption,
        }
        
        return {
            "nl_description": nl_description,  # Embed this
            "markdown": markdown,               # Pass to LLM
            "structured": structured,           # For programmatic access
        }
    
    def _table_to_nl(self, table, caption):
        """Convert table to natural language for embedding."""
        headers = table[0]
        rows = table[1:]
        desc = f"Table: {caption}. Columns: {', '.join(headers)}. "
        desc += f"Contains {len(rows)} rows. "
        # Add key stats
        for i, h in enumerate(headers):
            values = [row[i] for row in rows if i < len(row)]
            desc += f"{h} values include: {', '.join(values[:5])}. "
        return desc
    
    def _table_to_markdown(self, table, caption):
        """Convert to markdown for LLM context."""
        md = f"**{caption}**\n\n" if caption else ""
        md += "| " + " | ".join(table[0]) + " |\n"
        md += "| " + " | ".join(["---"] * len(table[0])) + " |\n"
        for row in table[1:]:
            md += "| " + " | ".join(row) + " |\n"
        return md
```

---

## 5. Audio RAG

```python
import openai

async def audio_to_rag_chunks(audio_path: str) -> list[dict]:
    """Transcribe audio → chunk → embed for RAG."""
    
    # 1. Transcribe with timestamps
    with open(audio_path, "rb") as f:
        transcript = await openai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    
    # 2. Create time-stamped chunks
    chunks = []
    for segment in transcript.segments:
        chunks.append({
            "text": segment["text"],
            "start_time": segment["start"],
            "end_time": segment["end"],
            "source": audio_path,
            "metadata": {
                "type": "audio",
                "timestamp": f"{segment['start']:.1f}s - {segment['end']:.1f}s",
            },
        })
    
    return chunks
    # When retrieved: show text + link to audio at specific timestamp
```

---

## 6. Text-to-SQL RAG

```mermaid
graph LR
    Q["User: 'Show top 10\ncustomers by revenue'"] --> SCHEMA["Inject DB Schema\ninto prompt"]
    SCHEMA --> LLM["LLM generates\nSQL query"]
    LLM --> VAL["Validate +\nSanitize SQL"]
    VAL --> DB["Execute on\nRead-only DB"]
    DB --> FMT["Format results\ninto answer"]
    FMT --> A["'Top customer is\nAcme Corp ($1.2M)'"]
```

```python
from openai import OpenAI
import sqlalchemy

# ── Schema Injection ──
SCHEMA = """
Tables:
  customers(id INT, name VARCHAR, email VARCHAR, created_at TIMESTAMP)
  orders(id INT, customer_id INT FK→customers, total DECIMAL, status VARCHAR, created_at TIMESTAMP)
  products(id INT, name VARCHAR, price DECIMAL, category VARCHAR)
  order_items(order_id INT FK→orders, product_id INT FK→products, quantity INT)
"""

def text_to_sql(question: str) -> str:
    """Generate SQL from natural language."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are a SQL expert. Given the schema below, 
generate a PostgreSQL query to answer the user's question.
Return ONLY the SQL query, no explanation.
IMPORTANT: Use only SELECT statements. Never UPDATE/DELETE/DROP.

{SCHEMA}"""},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip().strip("```sql").strip("```")

# ── Safety: Execute on Read-only Connection ──
engine = sqlalchemy.create_engine("postgresql://readonly_user:***@host/db")

def safe_execute(sql: str, max_rows: int = 100) -> list[dict]:
    """Execute SQL with safety checks."""
    sql_upper = sql.upper()
    
    # Block dangerous operations
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "EXEC"]
    if any(word in sql_upper for word in forbidden):
        raise ValueError(f"Dangerous SQL detected: {sql}")
    
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(sql + f" LIMIT {max_rows}"))
        return [dict(row._mapping) for row in result]

# Usage
query = text_to_sql("Show top 5 customers by total order value in 2024")
results = safe_execute(query)
```

```
When to use Text-to-SQL vs Vector RAG:
├── Text-to-SQL  → Structured data, aggregations, exact numbers
│   Examples: "Total revenue last quarter", "How many users signed up?"
├── Vector RAG   → Unstructured data, semantic search, documents  
│   Examples: "What's our refund policy?", "Find similar products"
└── Hybrid       → Both structured + unstructured data
    Example: "Which customers complained about shipping delays?" (SQL + docs)
```

---

## 🎯 Interview Tips — Chi Tiết

### Q1: "Multi-modal RAG challenges?"
**A**: (1) Parsing complex layouts (mixed text/tables/images). (2) Embedding different modalities in same space. (3) Cross-modal retrieval (text query → image result). (4) LLM must handle multi-modal context. (5) Cost of vision model calls during indexing.

### Q2: "PDF parsing best tool?"
**A**: Digital PDFs → PyMuPDF (fastest). Complex/scanned → Unstructured or Docling (best quality). Cloud API → LlamaParse (easiest). Academic papers → Docling (IBM, open-source, SOTA).

### Q3: "Table handling in RAG?"
**A**: Store 3 representations: (1) NL description for embedding/search. (2) Markdown for LLM context. (3) Structured for programmatic queries. Don't just embed raw table text — convert to descriptive sentences.

### Q4: "Image RAG pipeline?"
**A**: (1) Extract images from docs. (2) Vision model → text description. (3) Embed description. (4) Retrieve by text similarity. (5) Pass ORIGINAL image + description to multi-modal LLM for answer.

### Q5: "CLIP vs vision model description?"
**A**: CLIP: embed image directly, same space as text, fast, but limited to visual similarity. Vision model (GPT-4o): generates detailed text, richer search, but slower (one-time during indexing). Best: use both — CLIP for visual search, descriptions for semantic search.

### Q6: "Multimodal embedding models?"
**A**: CLIP (OpenAI): image+text, 512/768-dim, fast. SigLIP (Google): better than CLIP, more efficient training. Nomic Embed Vision: open-source, competitive. Jina CLIP v2: text+image in same space, multilingual. For production: Cohere multimodal embed or OpenAI embeddings with vision descriptions.
