# 🎯 AI Agents — Câu Hỏi Phỏng Vấn (30+)

---

## Agent Fundamentals (10 câu)

### Q1: AI Agent khác Chatbot thế nào?
**A**: Chatbot: react to messages (1 turn). Agent: autonomous multi-step reasoning + tool use + planning. Agent có loop: Think → Act → Observe → Think → ... → Final Answer.

### Q2: ReAct pattern giải thích?
**A**: Reasoning + Acting. Agent thinks (Thought), takes action (Action via tool), observes result (Observation), then thinks again. Iterates until has enough info for Final Answer. Paper: Yao et al. 2022.

### Q3: Function Calling vs ReAct?
**A**: Function Calling: native API (OpenAI/Anthropic), reliable JSON output, supports parallel tool calls. ReAct: text-based prompt engineering, works with any LLM, more flexible nhưng less reliable parsing.

### Q4: Planning strategies cho agents?
**A**: (1) ReAct: think-act-observe loop, good for <5 steps. (2) Plan-then-Execute: full plan → execute steps → replan if needed. (3) Tree-of-Thought: explore multiple paths. (4) Reflection: self-critique.

### Q5: Khi nào KHÔNG nên dùng agent?
**A**: Simple Q&A (dùng RAG), deterministic workflows (dùng code), latency-critical (<500ms), cost-sensitive (mỗi step = LLM call). Agents add complexity, cost, and unpredictability.

### Q6: Structured Output tại sao quan trọng cho agents?
**A**: Agent cần parse LLM output thành actions → unreliable nếu free-text. Structured output (Pydantic, JSON mode) → reliable parsing, type safety, validation.

### Q7: Agent error handling strategies?
**A**: (1) Retry with different params, (2) Fallback tools, (3) Escalate to human, (4) Timeout limits, (5) Max iterations, (6) Graceful degradation. Never let agent loop forever.

### Q8: Agent cost optimization?
**A**: (1) Use cheaper models (GPT-4o-mini) cho simple reasoning, (2) Cache tool results, (3) Limit max steps, (4) Batch tool calls, (5) Reduce prompt size, (6) Early termination.

### Q9: Agentic RAG vs Standard RAG?
**A**: Standard RAG: retrieve → generate (1 shot). Agentic RAG: agent decides WHEN to retrieve, WHAT to search, can do multiple searches, evaluate results, refine query. More accurate but slower.

### Q10: Agent observability?
**A**: Log every step: thoughts, actions, observations, latency, tokens. Trace visualization (LangSmith). Metrics: completion rate, steps/task, cost/task. Alerting on failures.

---

## LangGraph & Workflows (8 câu)

### Q11: LangGraph vs LangChain?
**A**: LangChain: sequential chains (A→B→C). LangGraph: directed graph with branches, loops, cycles, explicit state. LangGraph for complex workflows needing control flow.

### Q12: LangGraph State?
**A**: TypedDict defining shared data between nodes. Annotated fields with reducers (e.g., `Annotated[list, add]` for append-only lists). Passed through entire graph execution.

### Q13: Conditional Edges?
**A**: Route to different nodes based on state. Function evaluates state → returns route key → maps to next node. Enables if/else, loops, early termination.

### Q14: Checkpointing tại sao cần?
**A**: Save state at each node → resume after failure/interrupt. Enable: conversation memory (thread_id), human-in-the-loop (pause/resume), fault tolerance, debugging.

### Q15: Human-in-the-Loop trong LangGraph?
**A**: `interrupt_before=["dangerous_node"]`. Agent pauses, human reviews, approves/modifies state, then agent resumes. Essential cho critical actions (delete, payment, external API).

### Q16: Streaming trong LangGraph?
**A**: `astream_events()` — stream node completions + LLM tokens. User sees progress in real-time. Better UX than waiting for entire graph to complete.

### Q17: Supervisor pattern triển khai?
**A**: One supervisor node routes to specialist nodes (researcher, coder, writer). Supervisor sees all results, decides next step. Like a manager delegating to team.

### Q18: Map-Reduce trong agents?
**A**: Split task into N parallel subtasks → process each independently → merge results. Example: summarize 10 documents in parallel → combine summaries.

---

## Tools & MCP (6 câu)

### Q19: MCP Protocol là gì?
**A**: Model Context Protocol: chuẩn mở (Anthropic) cho AI-tool integration. Server expose tools + resources + prompts. Client (AI host) consume. "USB for AI tools" — write once, use everywhere.

### Q20: MCP components?
**A**: (1) Tools: functions agent can call, (2) Resources: data/context (files, DB schema), (3) Prompts: reusable prompt templates. Transport: stdio (local) or SSE (remote).

### Q21: Tool design best practices?
**A**: (1) Clear verb_noun naming, (2) Detailed descriptions, (3) Typed parameters with defaults, (4) Scoped responsibility, (5) Idempotent when possible, (6) Clear error messages.

### Q22: Parallel tool calls?
**A**: Modern models (GPT-4o) can call multiple tools simultaneously. Example: get_weather("Hanoi") AND get_weather("HCMC") in 1 turn. Handle: iterate over `msg.tool_calls` list.

### Q23: Tool vs Resource trong MCP?
**A**: Tool: action (has side effects, need params). Resource: data (read-only, identified by URI). Tool: search_database(query). Resource: config://database-schema.

### Q24: Tool error handling?
**A**: Return structured errors (not exceptions). Include: error type, message, suggestion. Agent can interpret and retry/fallback. Never crash the agent loop.

---

## Safety & Evaluation (6 câu)

### Q25: Agent safety risks?
**A**: (1) Prompt injection, (2) Tool misuse (delete data), (3) Excessive autonomy, (4) Hallucination in reasoning, (5) infinite loops, (6) PII leakage. Defense: guardrails, sandboxing, HITL.

### Q26: Prompt injection defense?
**A**: (1) Input validation (regex patterns), (2) Sandwich defense (system instructions wrap user input), (3) Output filtering, (4) Separate system/user contexts, (5) Content moderation API.

### Q27: Agent evaluation metrics?
**A**: (1) Task completion rate, (2) Tool accuracy (correct tool, correct params), (3) Steps to completion, (4) Latency, (5) Cost (tokens), (6) Hallucination rate, (7) User satisfaction.

### Q28: OWASP LLM Top 10?
**A**: Prompt Injection, Insecure Output, Data Poisoning, DoS, Supply Chain, Sensitive Info, Insecure Plugins, Excessive Agency, Overreliance, Model Theft. Key for production systems.

### Q29: Sandboxing code execution?
**A**: (1) Subprocess with timeout, (2) Banned operations whitelist, (3) Resource limits (CPU, memory), (4) No network access, (5) Temporary directories only. Docker for full isolation.

### Q30: Agent confidence routing?
**A**: High confidence (>0.9): auto-execute. Medium (0.7-0.9): execute with logging. Low (0.5-0.7): human review. Very low (<0.5): reject. Route based on model's self-assessment or heuristics.
