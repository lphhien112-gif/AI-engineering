# 🎯 Full-Stack AI — Câu Hỏi Phỏng Vấn (25+)

---

## API Design (5 câu)

### Q1: FastAPI vs Flask cho AI?
**A**: FastAPI: async, Pydantic validation, auto-docs (OpenAPI), type hints, faster (Starlette + uvicorn). Flask: simpler, larger ecosystem. AI/ML apps → FastAPI is standard.

### Q2: Pydantic validation?
**A**: Type safety + auto validation + serialization. Define Request/Response models with field constraints (min/max length, range). Auto-generates OpenAPI schema. Catches errors before reaching model.

### Q3: Background tasks khi nào?
**A**: Long-running: document processing, model training, batch inference. Pattern: POST → return job_id → client polls GET /jobs/{id}. Use Celery/Redis for production queues.

### Q4: API versioning strategy?
**A**: URL-based: /v1/predict, /v2/predict (most common). Header-based: X-API-Version. Always version ML APIs — models change, response schemas evolve.

### Q5: Error handling best practices?
**A**: Structured errors: {error, code, detail}. Use proper HTTP codes (400 input, 401 auth, 429 rate limit, 500 server). Log errors with request context. Never expose internal errors to clients.

---

## Streaming (4 câu)

### Q6: SSE vs WebSocket cho AI chat?
**A**: SSE: server→client only, HTTP-based, auto-reconnect, simpler. WebSocket: bidirectional, complex. Most AI chat = SSE (like ChatGPT). Voice/collab apps = WebSocket.

### Q7: Token streaming implementation?
**A**: Backend: async generator yields tokens as SSE events (`data: {json}\n\n`). Frontend: ReadableStream reader, append tokens to state. End signal: `data: [DONE]`.

### Q8: Buffering issues?
**A**: Nginx, CDN, proxy may buffer SSE responses. Fix: `X-Accel-Buffering: no`, `Cache-Control: no-cache`, `Transfer-Encoding: chunked`. Test full chain.

### Q9: Streaming error handling?
**A**: Mid-stream error → send error event → client shows partial + error. Connection drop → client auto-reconnects (SSE built-in). Timeout → server closes after max duration.

---

## Auth & Security (5 câu)

### Q10: JWT vs API Key?
**A**: JWT: user authentication, contains claims (role, email), expires (1-24h). API Key: service/machine auth, long-lived, simpler. Users → JWT, services → API Key.

### Q11: Rate limiting approaches?
**A**: Token bucket (smooth), sliding window (precise), fixed window (simple). Per-user + per-IP. Redis-based for distributed. Tiers: free (60/min), pro (600/min), enterprise (custom).

### Q12: Prompt injection defense?
**A**: (1) Input sanitization (regex patterns), (2) System prompt isolation, (3) Output filtering (PII, harmful content), (4) Separate user/system context, (5) Monitor for suspicious patterns.

### Q13: CORS configuration?
**A**: Whitelist specific origins: `allow_origins=["https://myapp.com"]`. Never `*` in production. Allow credentials only for trusted origins. Set proper headers for preflight requests.

### Q14: Secret management?
**A**: Never in code/git. Local: .env (gitignored). Cloud: Secret Manager (GCP/AWS). CI/CD: GitHub Secrets. Rotate regularly. Use different keys per environment.

---

## Frontend (4 câu)

### Q15: React chat UI patterns?
**A**: Message list (virtual scroll), input box (auto-resize), streaming cursor (blinking ▊), markdown rendering (react-markdown), code highlighting, copy button.

### Q16: Streaming state management?
**A**: Add empty assistant message → append tokens via setState. Use `useRef` for scroll-to-bottom. Avoid re-rendering entire message list (memoize messages).

### Q17: Vercel AI SDK advantages?
**A**: `useChat` hook: handles streaming, abort, retry, message state, loading. Multi-provider (OpenAI, Anthropic, etc.). Edge-compatible. Simplest way to build AI chat UI.

### Q18: Optimistic updates?
**A**: Show user message immediately (don't wait for server). Stream assistant response. On error: show error in-place, keep user message. User feels instant response.

---

## State Management (3 câu)

### Q19: Chat state design?
**A**: Conversations list, activeConversation, messages per conversation. Zustand: lightweight, persistent. Redux: overkill for most chat apps. Key: separate UI state from data state.

### Q20: Persistence strategies?
**A**: LocalStorage: simple, 5MB limit, same device. IndexedDB: large data, attachments. Supabase/Firebase: cross-device sync, real-time. Choose based on requirements.

### Q21: Conversation management?
**A**: Auto-title (LLM generates title from first message). Fork conversations. Search through history. Export/import. Archive old conversations.

---

## Deployment (4 câu)

### Q22: Full-stack AI deployment stack?
**A**: Frontend: Vercel (Next.js) or Netlify. Backend: Cloud Run / Railway / Modal. DB: Supabase / PlanetScale. Cache: Redis (Upstash). Monitoring: Sentry + custom dashboards.

### Q23: Cold start mitigation?
**A**: Set min-instances=1 (Cloud Run). Pre-warm models on startup. Use health check for readiness. Cache frequently used models in memory. Consider always-on for critical paths.

### Q24: Cost optimization?
**A**: (1) Scale-to-zero for dev/staging, (2) Cache LLM responses (Redis), (3) Use cheaper models for simple queries, (4) Batch requests where possible, (5) CDN for static assets.

### Q25: CI/CD for AI apps?
**A**: GitHub Actions: test → build → deploy. Backend: Cloud Run (auto-deploy on push). Frontend: Vercel (git integration). Include: model quality tests, API contract tests, E2E tests.
