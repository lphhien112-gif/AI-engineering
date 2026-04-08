# 🎯 AI Agents — Câu Hỏi Phỏng Vấn (40+)

> Mỗi câu quan trọng có: giải thích → code/pseudocode → trade-offs → follow-up.

---

## Agent Fundamentals (10 câu)

### Q1: AI Agent khác Chatbot thế nào?
**A**: 
- **Chatbot**: reactive, 1 turn, no memory between turns, no tools
- **Agent**: autonomous multi-step reasoning + tool use + planning + memory

```
Chatbot:   User → LLM → Response (done)
Agent:     User → Think → Act (tool) → Observe → Think → Act → ... → Final Answer
```

Key difference: Agent has **agency** — decides WHAT to do, WHEN to stop, WHICH tools to use. Chatbot chỉ respond.

### Q2: ReAct pattern giải thích?
**A**: **Re**asoning + **Act**ing. Agent interleaves thinking with tool use.

```python
# ReAct loop pseudocode
def react_agent(question: str, tools: list, max_steps: int = 5):
    messages = [{"role": "user", "content": question}]
    
    for step in range(max_steps):
        # 1. THINK — LLM reasons about what to do
        response = llm.chat(messages)  
        
        # 2. Check if agent wants to use a tool
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # 3. ACT — execute tool
                result = execute_tool(tool_call.name, tool_call.args)
                # 4. OBSERVE — feed result back
                messages.append({"role": "tool", "content": result})
        else:
            return response.content  # Final answer — no more tools needed
    
    return "Max steps reached"
```
Paper: Yao et al. 2022. Strength: simple, effective for < 5 steps. Weakness: can loop on hard problems.

### Q3: Function Calling vs ReAct?
**A**: 

| | Function Calling | ReAct (prompt-based) |
|-|-----------------|---------------------|
| **Implementation** | Native API (OpenAI, Anthropic) | Prompt engineering |
| **Output format** | Reliable JSON | Text parsing (fragile) |
| **Parallel tools** | ✅ natively | ❌ sequential only |
| **Model support** | Specific models | Any LLM |
| **Flexibility** | Fixed schema | More creative reasoning |

**2026 standard**: Function Calling cho production. ReAct cho research/prototyping.

### Q4: Planning strategies cho agents?
**A**: 

| Strategy | Cách hoạt động | Khi nào dùng | Trade-off |
|----------|---------------|-------------|-----------|
| **ReAct** | Think→Act→Observe loop | Simple tasks, < 5 steps | Fast but can loop |
| **Plan-then-Execute** | Full plan upfront → execute steps | Complex multi-step | Better structure, but rigid plan |
| **Reflexion** | Execute → self-critique → retry | Tasks needing iteration | Slower, more tokens, higher quality |
| **Tree-of-Thought** | Explore multiple paths | Ambiguous problems | Expensive (multiple branches) |

- **Follow-up**: "Plan-then-Execute handle changes thế nào?" → Replan after each step if needed (dynamic replanning). Or use LATS (Language Agent Tree Search) for backtracking.

### Q5: Khi nào KHÔNG nên dùng agent?
**A**: 
- **Simple Q&A** → RAG (deterministic, cheaper)
- **Deterministic workflows** → code/scripts (reliable, testable)
- **Latency-critical** (< 500ms) → agents add multiple LLM calls
- **Cost-sensitive** → mỗi step = 1 LLM call, 5-step agent = 5× cost
- **Rule of thumb**: If you can write an `if/else` for it, don't use an agent.

### Q6: Structured Output tại sao quan trọng cho agents?
**A**: Agent cần parse LLM output thành actions. Free-text → unreliable parsing → agent crashes.

```python
from pydantic import BaseModel
from openai import OpenAI

class AgentAction(BaseModel):
    thought: str
    tool_name: str
    tool_args: dict

# Structured output → 100% valid JSON, type-safe
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    response_format=AgentAction,
    messages=[...],
)
action = response.choices[0].message.parsed  # AgentAction object
```

### Q7: Agent error handling strategies?
**A**: 6 strategies (ordered by escalation):
1. **Retry** with different params / rephrased query
2. **Fallback** to alternative tool (search_web → search_docs)
3. **Self-correct** — ask LLM to fix its own error
4. **Skip** and continue (if tool result optional)
5. **Escalate** to human (HITL)
6. **Graceful termination** with partial results

**⚠️ Golden rule**: set `max_iterations` and `timeout`. Never let agent loop forever.

### Q8: Agent cost optimization?
**A**: 
1. **Tiered models**: GPT-4o-mini cho routing/simple reasoning, GPT-4o cho complex only
2. **Cache tool results**: same search query → cached response (Redis, TTL 1h)
3. **Limit max steps**: 5-10 steps max, early termination if confidence high
4. **Prompt compression**: shorter system prompt, remove unnecessary context
5. **Batch related questions**: 1 agent call xử lý 3 related questions thay vì 3 calls

### Q9: Agentic RAG vs Standard RAG?
**A**: 
```
Standard RAG:  query → search(1x) → generate answer  (1 retrieval)
Agentic RAG:   query → think → search → evaluate → 
               "not enough info" → refine query → search again → 
               merge results → generate answer  (multi-retrieval)
```
- Agent **decides** when to retrieve, what to search, can do multiple searches, evaluate result quality
- **When**: complex questions needing multiple sources, when first retrieval often insufficient
- **Trade-off**: 2-5× slower & costlier, but significantly better recall on hard questions

### Q10: Agent observability?
**A**: Log **every step**:
- Step N: thought, action (tool + args), observation, latency, tokens used
- End-to-end: total steps, total tokens, total cost, success/failure

**Tools**: LangSmith (trace visualization), AgentOps, Langfuse (open-source)
**Metrics**: completion rate, avg steps/task, cost/task, tool error rate

---

## LangGraph & Workflows (8 câu)

### Q11: LangGraph vs LangChain?
**A**: 
- **LangChain**: sequential chains (`chain = prompt | llm | parser`). Linear A→B→C
- **LangGraph**: directed graph — branches, loops, cycles, conditional routing, explicit state

```python
# LangChain — linear
chain = prompt | llm | output_parser
result = chain.invoke({"query": "..."})

# LangGraph — graph with control flow
from langgraph.graph import StateGraph

graph = StateGraph(AgentState)
graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_conditional_edges("research", should_continue,
    {"continue": "research", "finish": "write"})  # Loop!
```
- **When LangGraph**: complex workflows needing loops, branching, human-in-the-loop, state management.

### Q12: LangGraph State?
**A**: TypedDict defining shared data between nodes. Annotated fields with **reducers** control how state updates merge.
```python
from typing import Annotated, TypedDict
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]     # Append-only (each node adds messages)
    current_step: str                   # Overwrite (last write wins)
    search_results: Annotated[list, add]  # Accumulate across nodes
```
State is **immutable per node** — each node receives state, returns updates, reducer merges.

### Q13: Conditional Edges?
**A**: Route to different nodes based on state.
```python
def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tool_node"      # Agent wants to use tools
    return "end"                # Agent finished

graph.add_conditional_edges("agent", should_continue, 
    {"tool_node": "tools", "end": END})
```
Enables: if/else branching, loops (node → condition → same node), early termination.

### Q14: Checkpointing tại sao cần?
**A**: Save state at each node → resume after failure/interrupt.
- **Conversation memory**: `thread_id` → restore previous conversation
- **Human-in-the-loop**: pause → human reviews → resume
- **Fault tolerance**: crash at step 5 → resume from step 5 (not restart)
- **Debugging**: replay execution from any checkpoint
- **Storage**: SQLite (dev), PostgreSQL (prod)

### Q15: Human-in-the-Loop trong LangGraph?
**A**: `interrupt_before=["dangerous_node"]`. Agent pauses, human reviews state, approves/modifies, then resumes.
- Essential for: delete operations, payments, external API calls, sending emails
- UX pattern: agent proposes action → human clicks approve/reject → agent continues or replans

### Q16: Streaming trong LangGraph?
**A**: `astream_events()` — stream node completions + LLM tokens. User sees real-time progress.
- Stream **node transitions**: "Researching..." → "Analyzing..." → "Writing..."
- Stream **LLM tokens**: within each node, token-by-token output
- Better UX than waiting 30s for entire graph to complete.

### Q17: Supervisor pattern?
**A**: One supervisor LLM routes to specialist nodes (researcher, coder, writer).
- Supervisor sees question → decides "this needs research" → routes to researcher node
- Researcher returns results → supervisor decides "now needs code" → routes to coder
- Like a manager delegating tasks to team members

### Q18: Map-Reduce trong agents?
**A**: Split task → process each independently (parallel) → merge results.
- **Example**: "Summarize 10 documents" → 10 parallel summarize calls → 1 merge call
- **Benefit**: parallelism, each sub-task simpler
- **LangGraph**: use `Send()` API to fan out to multiple node instances

---

## Tools & MCP (8 câu)

### Q19: MCP (Model Context Protocol) là gì?
**A**: Open standard (Anthropic) cho AI-tool integration. "USB-C for AI" — write tool once, works with any AI host.

```
Architecture:
  AI Host (Claude, Cursor, etc.)
    ↕ MCP Client
    ↕ MCP Protocol (JSON-RPC over stdio/SSE)
    ↕ MCP Server
    ↕ Your tools, data, services
```

### Q20: MCP components?
**A**: 3 primitives:
- **Tools**: functions agent can call (side effects OK). VD: `search_db(query)`
- **Resources**: read-only data (no side effects). VD: `config://database-schema`
- **Prompts**: reusable prompt templates. VD: `summarize(style="technical")`

**Transport**: stdio (local processes), SSE/HTTP (remote servers). Stdio = most common for desktop tools.

### Q21: Tool design best practices?
**A**: 
1. **Naming**: `verb_noun` — `search_documents`, `create_ticket` (LLM understands intent)
2. **Description**: detailed — "Search product catalog by name, category. Returns top 10 matches with price." (LLM READS this to decide when to use)
3. **Parameters**: typed + constrained + defaults + examples in description
4. **Scope**: single responsibility. `search_and_update_and_notify` = BAD
5. **Idempotent**: when possible (safe to retry)
6. **Error messages**: clear, actionable — "User not found. Try search_users first." (LLM can self-correct)

### Q22: Parallel tool calls?
**A**: Modern models (GPT-4o, Claude 3.5) call multiple tools simultaneously in 1 turn.
```python
# Model response may contain multiple tool_calls
for tool_call in response.tool_calls:
    result = execute_tool(tool_call.function.name, 
                          json.loads(tool_call.function.arguments))
    # Collect all results → send back in 1 message
```
Benefit: `get_weather("Hanoi") + get_weather("HCMC")` in 1 round-trip instead of 2.

### Q23: Tool vs Resource trong MCP?
**A**: 
- **Tool**: action-oriented, has side effects, needs parameters. `search_database(query="AI")`
- **Resource**: data-oriented, read-only, identified by URI. `file:///path/to/schema.sql`
- **Analogy**: Tool = POST endpoint. Resource = GET endpoint.

### Q24: Tool error handling?
**A**: Return structured errors (not exceptions). Agent can interpret and retry/fallback.
```python
def search_tool(query: str) -> str:
    try:
        results = db.search(query)
        return json.dumps({"status": "success", "results": results})
    except ConnectionError:
        return json.dumps({
            "status": "error",
            "error_type": "connection_failed",
            "message": "Database unavailable. Try again in 30s.",
            "suggestion": "Use cached_search tool as fallback."
        })
```
**Rule**: Never crash the agent loop. Always return a parseable response.

### Q25: Dynamic tool selection?
**A**: Agent có nhiều tools nhưng mỗi query chỉ cần subset. Vấn đề: 50 tools = huge prompt.
- **Solution 1**: Tool retrieval — embed tool descriptions, retrieve top-K relevant tools per query
- **Solution 2**: Category routing — classify query → load category-specific tools
- **Solution 3**: Two-stage — planner selects tools → executor uses selected tools

### Q26: Tool composition (chaining)?
**A**: Complex tasks need multiple tools in sequence.
```
"Find highest-paid employees in each department"
→ Step 1: list_departments() → ["Engineering", "Sales", ...]
→ Step 2: for each dept → query_employees(dept, sort="salary", limit=1)
→ Step 3: format_table(results)
```
Agent learns to compose autonomously. Key: each tool returns structured data the next tool can consume.

---

## Safety & Evaluation (8 câu)

### Q25: Agent safety risks?
**A**: 6 categories:
1. **Prompt injection**: user manipulates agent via crafted input
2. **Tool misuse**: agent calls `delete_all_users()` unintentionally
3. **Excessive autonomy**: agent takes actions beyond scope
4. **Reasoning hallucination**: confident but wrong reasoning chain
5. **Infinite loops**: agent stuck, burning tokens  
6. **Data leakage**: agent exposes PII or system prompts

### Q26: Prompt injection defense?
**A**: Defense in depth:
1. **Input validation**: regex patterns, content moderation API
2. **Sandwich defense**: system instructions wrap user input
3. **Tool permissions**: allowlist (not blocklist) of allowed actions
4. **Output filtering**: check agent output before executing
5. **Monitoring**: detect anomalous tool call patterns
- **Follow-up**: "Indirect injection?" → Malicious content in retrieved documents (RAG) tricks agent. Defense: treat retrieved content as untrusted, separate system/user/retrieved contexts.

### Q27: Agent evaluation metrics?
**A**: 

| Metric | Measures | How |
|--------|----------|-----|
| **Task completion rate** | Can agent finish task? | % of test cases solved correctly |
| **Tool accuracy** | Right tool + right params? | Compare vs golden tool sequence |
| **Steps to completion** | Efficiency | Avg steps (fewer = better) |
| **Cost per task** | Token efficiency | Total tokens × pricing |
| **Hallucination rate** | Factual grounding | % answers not supported by tool outputs |
| **Safety violations** | Guardrail effectiveness | % attempts that triggered safety rules |

### Q28: Agent benchmarks?
**A**: 
- **SWE-bench**: Can agent fix real GitHub issues? (code generation + debugging)
- **ToolBench**: Can agent use 16K+ real APIs correctly?
- **WebArena**: Can agent navigate websites to complete tasks?
- **GAIA**: General AI Assistant benchmark (multi-step reasoning + tools)
- **Custom**: build dataset specific to your use case (most important for production)

### Q29: OWASP LLM Top 10?
**A**: LLM01-Prompt Injection, LLM02-Insecure Output, LLM03-Training Data Poisoning, LLM04-Model DoS, LLM05-Supply Chain, LLM06-Sensitive Info Disclosure, LLM07-Insecure Plugins, LLM08-Excessive Agency, LLM09-Overreliance, LLM10-Model Theft.
- Most critical for agents: **LLM07** (insecure plugins/tools) + **LLM08** (excessive agency).

### Q30: Sandboxing code execution?
**A**: Agent generates code → NEVER run directly. Sandbox layers:
1. Subprocess with timeout (kill after 30s)
2. Banned operations whitelist (no `os.system`, `import subprocess`)
3. Resource limits (CPU, memory caps)
4. No network access (isolated)
5. Temporary directories only (no filesystem access)
6. **Docker** for full isolation (best practice)

### Q31: Agent confidence routing?
**A**: Route based on agent's self-assessed confidence:
- **High (> 0.9)**: auto-execute
- **Medium (0.7-0.9)**: execute with logging
- **Low (0.5-0.7)**: human review required
- **Very low (< 0.5)**: reject, ask user to rephrase
- Implementation: ask LLM to self-rate confidence, or use heuristics (steps taken, tool errors).

### Q32: Multi-agent coordination?
**A**: Patterns for multiple agents working together:
- **Supervisor**: one agent delegates to specialist agents
- **Debate**: 2+ agents argue, converge on answer (better reasoning)
- **Pipeline**: agent A output → agent B input → agent C (sequential)
- **Swarm**: agents negotiate, share tasks dynamically
- **Communication**: shared state (memory) vs message passing. Shared state simpler, message passing scales better.
- **Follow-up**: "OpenAI Agents SDK?" → Formerly Swarm. Lightweight multi-agent framework. Handoff between agents, tool routing. Good for customer service flows.
