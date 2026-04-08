# 🎯 Full-Stack AI — Câu Hỏi Phỏng Vấn (35+)

> Mỗi câu quan trọng có: giải thích → production pattern → edge cases → follow-up.

---

## API Design (5 câu)

### Q1: FastAPI vs Flask cho AI?
**A**: 

| | FastAPI | Flask |
|-|---------|-------|
| **Async** | Native async/await | Sync (async via extensions) |
| **Validation** | Built-in Pydantic | Manual / Flask-Marshmallow |
| **API docs** | Auto OpenAPI + Swagger | Manual / Flask-RESTX |
| **Performance** | ~2-3× faster (Starlette + uvicorn) | WSGI, slower |
| **Type hints** | First-class citizen | Optional |
| **Learning curve** | Moderate | Easy |

**2026 standard**: AI/ML APIs → FastAPI. Legacy web apps → Flask/Django.
- **Follow-up**: "Litestar?" → Newer alternative to FastAPI. Similar features, better OpenAPI spec compliance. Consider for new projects.

### Q2: Pydantic validation — tại sao critical cho AI APIs?
**A**: AI inputs are messy. Without validation = garbage in, garbage out.
```python
from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    messages: list[dict] = Field(..., min_length=1, max_length=100)
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=128000)
    stream: bool = Field(default=False)
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        for msg in v:
            if msg.get("role") not in ("user", "assistant", "system"):
                raise ValueError(f"Invalid role: {msg['role']}")
        return v
```
- Auto-generates OpenAPI schema → client SDKs auto-generated
- Catches errors **before** reaching expensive LLM call (save cost!)

### Q3: Background tasks khi nào?
**A**: Long-running operations → don't block API response.
```python
from fastapi import BackgroundTasks

@app.post("/process-document")
async def process_doc(file: UploadFile, bg: BackgroundTasks):
    job_id = str(uuid4())
    bg.add_task(process_pdf, job_id, file)  # Runs after response
    return {"job_id": job_id, "status": "processing"}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    return job_store.get(job_id)  # Client polls this
```
- **Simple jobs**: FastAPI `BackgroundTasks`
- **Production queue**: Celery + Redis (persistent, retry, monitoring)
- **Serverless**: Cloud Run Jobs, AWS Lambda

### Q4: API versioning strategy?
**A**: URL-based most common: `/v1/chat`, `/v2/chat`.
- **Why for AI**: model response format changes, new capabilities, deprecating old models
- **Deprecation policy**: announce 3 months ahead, sunset date in response headers
- **⚠️ Don't**: version every small change. Only when breaking backward compatibility.

### Q5: Error handling best practices?
**A**: 
```python
# Unified error format
class APIError(BaseModel):
    error: str          # "rate_limit_exceeded"
    message: str        # Human-readable description
    code: int           # HTTP status code
    detail: dict = {}   # Extra context

# Mapping
HTTP_CODES = {
    400: "Bad request (invalid input)",
    401: "Unauthorized (missing/invalid API key)",
    429: "Rate limited (slow down!)",
    500: "Internal error (our fault)",
    503: "Model unavailable (retry later)",
}
```
- **⚠️ Never**: expose Python tracebacks, model internals, or raw LLM errors to users.

---

## Streaming (5 câu)

### Q6: SSE vs WebSocket cho AI chat?
**A**: 

| | SSE | WebSocket |
|-|-----|-----------|
| **Direction** | Server → Client only | Bidirectional |
| **Protocol** | HTTP (standard) | Upgrade to WS protocol |
| **Reconnect** | Built-in auto-reconnect | Manual implementation |
| **Complexity** | Simple | Complex |
| **Proxy/CDN** | Some buffering issues | Generally works |
| **Use case** | Chat streaming (ChatGPT-like) | Voice, collab editing, gaming |

**Decision**: 95% of AI chat apps → SSE (simpler, sufficient). Voice apps → WebSocket.
- ChatGPT, Claude, Gemini all use SSE for text streaming.

### Q7: Token streaming implementation?
**A**: 
```python
# Backend (FastAPI)
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for chunk in openai_client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            stream=True,
        ):
            token = chunk.choices[0].delta.content or ""
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering!
        },
    )
```

```javascript
// Frontend (JavaScript)
const response = await fetch("/chat/stream", {
  method: "POST",
  body: JSON.stringify({ messages, stream: true }),
  signal: abortController.signal,  // Cancel support
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const text = decoder.decode(value);
  for (const line of text.split("\n")) {
    if (line.startsWith("data: ") && line !== "data: [DONE]") {
      const { token } = JSON.parse(line.slice(6));
      appendToken(token);  // Update UI
    }
  }
}
```

### Q8: Streaming edge cases?
**A**: Production streaming breaks in surprising ways:
1. **Nginx buffering**: responses batch up → user sees nothing then everything. Fix: `X-Accel-Buffering: no`
2. **CDN buffering**: Cloudflare, AWS ALB may buffer. Disable response buffering.
3. **Connection drop mid-stream**: server keeps generating (wasting tokens). Fix: detect disconnect, cancel upstream.
4. **Partial JSON in stream**: structured output arrives token-by-token → invalid JSON until complete. Fix: buffer until valid JSON or use Instructor streaming.
5. **Unicode split across chunks**: multi-byte chars (emoji, CJK) split between chunks. Fix: TextDecoder with `stream: true`.
6. **AbortController**: user clicks "Stop generating" → signal abort to server.

```python
# Server-side: detect client disconnect
async def generate():
    async for chunk in llm.stream(messages):
        if await request.is_disconnected():
            break  # Stop wasting tokens!
        yield f"data: {chunk}\n\n"
```

### Q9: Streaming error handling?
**A**: Errors can happen DURING stream (not just before):
- **Before stream**: return normal HTTP error (400, 500)
- **During stream**: send error event → client shows partial response + error
```python
async def generate():
    try:
        async for chunk in llm.stream(messages):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
    except OpenAIError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"
```
- Client: if receives error event → show partial + "Generation interrupted" message

### Q10: Token counting + cost estimation?
**A**: 
```python
import tiktoken

def count_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += 4  # Message overhead
        total += len(enc.encode(msg["content"]))
    total += 2  # Reply priming
    return total

def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    prices = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}
    inp, out = prices[model]
    return (input_tokens * inp + output_tokens * out) / 1_000_000
```
- **Pre-request**: count input tokens → reject if exceeds budget
- **Post-request**: log actual usage → track cost per user/feature

---

## Auth & Security (5 câu)

### Q11: JWT vs API Key vs OAuth2?
**A**: 

| | JWT | API Key | OAuth2 |
|-|-----|---------|--------|
| **For** | User authentication | Service/machine auth | Third-party access |
| **Lifetime** | Short (15min-24h) | Long-lived | Token + refresh |
| **Claims** | Encoded (role, email) | Just identifier | Scoped permissions |
| **Stateless** | ✅ (verify signature) | ❌ (lookup in DB) | Depends |
| **Rotation** | Built-in (expiry) | Manual | Built-in |

**Decision**: Users → JWT. Service-to-service → API Key. Third-party → OAuth2.

**JWT security pitfalls**:
- Algorithm confusion (always specify `algorithms=["HS256"]`)
- No blacklist (can't invalidate before expiry — use short expiry + refresh token)
- Payload not encrypted (don't store PII in JWT — only in encrypted claims)

### Q12: Rate limiting approaches?
**A**: 

| Algorithm | Behavior | Best for |
|-----------|----------|----------|
| **Token bucket** | Smooth, allows bursts | API rate limiting |
| **Sliding window** | Precise, no burst | Strict limits |
| **Fixed window** | Simple, edge spike | Basic protection |

```python
# Production: per-user rate limiting with Redis
from fastapi import Request
import redis

async def rate_limit(request: Request, limit: int = 60, window: int = 60):
    user_key = f"rate:{request.state.user_id}"
    current = redis_client.incr(user_key)
    if current == 1:
        redis_client.expire(user_key, window)
    if current > limit:
        raise HTTPException(429, "Rate limit exceeded. Try again later.")
```
- **Tiers**: free (10 req/min), pro (100 req/min), enterprise (custom)
- **Multiple layers**: Nginx/CDN (volumetric DDoS), app-level (per-user)

### Q13: Prompt injection defense?
**A**: Defense in depth — no single solution:
1. **Input sanitization**: regex filter known attack patterns
2. **System prompt isolation**: separate system/user in API structure
3. **Output filtering**: check for PII, harmful content before returning
4. **Context separation**: retrieved docs (RAG) treated as untrusted
5. **Monitoring**: detect anomalous output patterns, flag for review
- **Follow-up**: "Indirect injection?" → Malicious instructions embedded in documents/websites that agent retrieves → agent follows attacker's instructions. Hardest to defend.

### Q14: CORS configuration?
**A**: 
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.com"],  # NEVER ["*"] in production!
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```
- **Development**: `allow_origins=["http://localhost:3000"]`
- **Production**: whitelist specific domains only
- **⚠️ Common bug**: CORS errors on streaming SSE — ensure `allow_headers` includes all custom headers.

### Q15: Secret management?
**A**: 
- **Local dev**: `.env` file (`.gitignore`'d) + `python-dotenv`
- **CI/CD**: GitHub Secrets, GitLab CI Variables
- **Production**: GCP Secret Manager, AWS Secrets Manager, HashiCorp Vault
- **Rotation**: API keys rotated quarterly minimum. LLM API keys: monthly.
- **⚠️ Never**: commit to git. Even if you delete it, it's in git history forever. Use `git-secrets` pre-commit hook.

---

## Frontend (5 câu)

### Q16: React chat UI architecture?
**A**: 
```
Component tree:
  ChatApp
  ├── Sidebar (conversation list, search)
  ├── ChatWindow
  │   ├── MessageList (virtual scroll for performance)
  │   │   ├── UserMessage
  │   │   └── AssistantMessage (markdown + code highlight)
  │   ├── StreamingCursor (blinking ▊ during generation)
  │   └── InputBox (auto-resize textarea, send button)
  └── SettingsPanel (model, temperature, system prompt)
```

Key patterns:
- **Virtual scrolling**: `react-virtuoso` for conversations with 1000+ messages
- **Markdown rendering**: `react-markdown` + `rehype-highlight` for code blocks
- **Copy button**: per code block, one-click copy
- **Auto-scroll**: scroll to bottom on new messages, stop if user scrolls up

### Q17: Streaming state management?
**A**: 
```javascript
// Pattern: add empty assistant message → append tokens
const [messages, setMessages] = useState([]);
const [isStreaming, setIsStreaming] = useState(false);

async function sendMessage(content) {
  // 1. Add user message immediately (optimistic)
  setMessages(prev => [...prev, { role: "user", content }]);
  
  // 2. Add empty assistant message
  const assistantId = uuid();
  setMessages(prev => [...prev, { id: assistantId, role: "assistant", content: "" }]);
  setIsStreaming(true);
  
  // 3. Stream tokens → append to assistant message
  for await (const token of streamChat(content)) {
    setMessages(prev => prev.map(msg => 
      msg.id === assistantId 
        ? { ...msg, content: msg.content + token }
        : msg
    ));
  }
  setIsStreaming(false);
}
```
- **⚠️ Performance**: `setMessages` per token = many re-renders. Use `useRef` for content accumulation, `requestAnimationFrame` for batched UI updates.

### Q18: Vercel AI SDK?
**A**: `useChat` hook handles streaming, abort, retry, message state, loading — all in one hook.
```javascript
import { useChat } from 'ai/react';

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit, 
          isLoading, stop, reload } = useChat({
    api: '/api/chat',
  });
  
  return (
    <div>
      {messages.map(m => <Message key={m.id} {...m} />)}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
        {isLoading && <button onClick={stop}>Stop</button>}
      </form>
    </div>
  );
}
```
- Multi-provider: OpenAI, Anthropic, Google. Edge-compatible.
- **When NOT to use**: need custom streaming logic, WebSocket, or non-React framework.

### Q19: Optimistic updates?
**A**: Show user message **immediately** (don't wait for server confirmation).
- Stream assistant response in real-time
- On error: keep user message visible, show error inline
- **Result**: app feels instant. User types → message appears → AI starts responding within 200ms.

### Q20: Accessibility in AI chat?
**A**: 
- **Screen readers**: aria-live="polite" on streaming messages (announce new content)
- **Keyboard navigation**: Tab to navigate, Enter to send, Escape to stop
- **High contrast**: ensure readable markdown in both light/dark themes
- **Focus management**: auto-focus input after send, skip to latest message

---

## State Management (4 câu)

### Q21: Chat state design?
**A**: 
```javascript
// Zustand store — lightweight, persistent
const useChatStore = create(
  persist(
    (set, get) => ({
      conversations: {},           // { id: { title, messages, createdAt } }
      activeConversationId: null,
      
      addMessage: (convId, message) => set(state => ({
        conversations: {
          ...state.conversations,
          [convId]: {
            ...state.conversations[convId],
            messages: [...state.conversations[convId].messages, message],
          }
        }
      })),
      
      deleteConversation: (id) => set(state => {
        const { [id]: _, ...rest } = state.conversations;
        return { conversations: rest };
      }),
    }),
    { name: 'chat-storage', storage: createJSONStorage(() => localStorage) }
  )
);
```

### Q22: Undo/redo pattern cho chat?
**A**: Command pattern — regenerate response, restore previous version.
```javascript
// Track history for undo
const useUndoStore = create((set, get) => ({
  past: [],      // Previous states
  present: null, // Current state
  future: [],    // Redo stack
  
  regenerate: (messageId) => {
    const current = get().present;
    set(state => ({
      past: [...state.past, current],
      present: removeMessageAndAfter(current, messageId),
      future: [],  // Clear redo on new action
    }));
    // Trigger new AI generation...
  },
  
  undo: () => set(state => ({
    past: state.past.slice(0, -1),
    present: state.past[state.past.length - 1],
    future: [state.present, ...state.future],
  })),
}));
```

### Q23: Persistence strategies?
**A**: 

| Storage | Capacity | Sync | Best for |
|---------|:--------:|:----:|----------|
| **localStorage** | 5-10MB | Same device | Quick prototype |
| **IndexedDB** | 50-250MB | Same device | Large chat history, attachments |
| **Supabase/Firebase** | Unlimited | Cross-device, real-time | Multi-device apps |
| **Custom API** | Unlimited | Cross-device | Full control, enterprise |

### Q24: Offline-first pattern?
**A**: Queue messages when offline → sync when back online.
```javascript
const offlineQueue = [];

async function sendMessage(content) {
  if (!navigator.onLine) {
    offlineQueue.push({ content, timestamp: Date.now() });
    showToast("Saved offline. Will send when back online.");
    return;
  }
  await actualSend(content);
}

window.addEventListener('online', async () => {
  for (const msg of offlineQueue) {
    await actualSend(msg.content);
  }
  offlineQueue.length = 0;
  showToast("Messages synced!");
});
```

---

## Deployment (4 câu)

### Q25: Full-stack AI deployment stack?
**A**: 

| Layer | Option 1 (Simple) | Option 2 (Scalable) |
|-------|-------------------|---------------------|
| **Frontend** | Vercel (Next.js) | AWS CloudFront + S3 |
| **Backend** | Cloud Run | K8s + autoscaler |
| **Database** | Supabase | PostgreSQL + pgvector |
| **Cache** | Upstash Redis | ElastiCache Redis |
| **Vector DB** | Qdrant Cloud | Self-hosted Qdrant |
| **Monitoring** | Sentry + Langfuse | Datadog + custom |

### Q26: Cold start mitigation?
**A**: 
- **Min instances**: Cloud Run `--min-instances=1` (keeps 1 warm)
- **Pre-warm**: load model on startup, run dummy inference
- **Health check**: readiness probe verifies model loaded before receiving traffic

```python
@app.on_event("startup")
async def startup():
    global model
    model = load_model("model.onnx")      # Load during startup
    model.predict(dummy_input)             # Warm up (compile kernels)
    logger.info("Model loaded and warmed up")
```

### Q27: Cost estimation cho AI app?
**A**: 
```
Monthly cost for 1000 DAU, 10 messages/user/day:

LLM API:     10K msgs × $0.003/msg (GPT-4o-mini) = $30/mo
Hosting:     Cloud Run (always-on) = $50/mo
Database:    Supabase free tier = $0/mo
Vector DB:   Qdrant Cloud starter = $25/mo
Redis:       Upstash free tier = $0/mo
Domain/SSL:  $12/year
─────────────────────────────
Total:       ~$105/mo at 1K DAU

Scaling:
  10K DAU:   ~$400/mo (LLM cost dominates)
  100K DAU:  ~$3,500/mo (need caching, model routing)
```
- **Key insight**: LLM API cost scales linearly with usage. Everything else has good free/cheap tiers at small scale.

### Q28: CI/CD for AI apps?
**A**: 
```yaml
# GitHub Actions
name: Deploy AI App
on:
  push:
    branches: [main]
jobs:
  test:
    steps:
      - run: pytest tests/ -v
      - run: python tests/test_prompts.py      # Test prompt templates
  deploy-backend:
    needs: test
    steps:
      - uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: ai-api
          image: gcr.io/$PROJECT/ai-api:$GITHUB_SHA
  deploy-frontend:
    needs: test
    # Vercel auto-deploys on push (git integration)
```
- Include: model quality tests, API contract tests, prompt regression tests

---

## Database & Architecture (3 câu) 🆕

### Q29: Database schema cho chat apps?
**A**: 
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    title TEXT,  -- Auto-generated from first message
    model TEXT DEFAULT 'gpt-4o',
    system_prompt TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    tokens_used INTEGER,       -- Track consumption
    model TEXT,                -- Which model generated this
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX idx_messages_conv ON messages(conversation_id, created_at);
CREATE INDEX idx_conv_user ON conversations(user_id, updated_at DESC);
```

### Q30: Performance at scale?
**A**: 
- **Virtual scrolling**: only render visible messages (react-virtuoso). 10K messages = same performance as 10.
- **Pagination**: load last 50 messages → scroll up triggers load more
- **Lazy loading**: conversation list shows titles only → load messages on click
- **Search**: full-text search on messages (PostgreSQL tsvector / Elasticsearch)
- **Archival**: auto-archive conversations older than 6 months → cold storage

### Q31: Real-time features beyond chat?
**A**: WebSocket for:
- **Presence**: show who's online / typing
- **Collaborative editing**: multiple users editing same prompt/document
- **Live dashboard**: real-time metrics, active users, cost ticker
- **Notifications**: push alerts for long-running jobs completing
- **Pattern**: WebSocket for real-time events, REST/SSE for request-response AI interactions
