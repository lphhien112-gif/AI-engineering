"""
📊 LangGraph Workflow Demo — State Machine Agent (no API key needed)
Chạy: python langgraph_workflow.py

Simulates a LangGraph-style workflow without requiring langgraph installed.
Shows: State management, Node execution, Conditional routing, Checkpointing.
"""
import json
from dataclasses import dataclass, field
from typing import Callable
from datetime import datetime
import time


# ============================================================
# State — shared data between nodes
# ============================================================

@dataclass
class WorkflowState:
    """Typed state object passed between nodes."""
    query: str = ""
    research_results: list[str] = field(default_factory=list)
    draft: str = ""
    review_feedback: str = ""
    final_output: str = ""
    step_count: int = 0
    history: list[str] = field(default_factory=list)
    status: str = "initialized"


# ============================================================
# Nodes — functions that modify state
# ============================================================

def researcher_node(state: WorkflowState) -> WorkflowState:
    """Research node: gather information about the query."""
    print(f"  🔍 Researcher: Searching for '{state.query}'...")
    time.sleep(0.3)
    
    # Simulated research
    knowledge_base = {
        "rag": [
            "RAG combines retrieval with generation for more accurate AI responses.",
            "Key components: Document chunking, Embedding, Vector DB, Retrieval, Generation.",
            "Best practices: Hybrid search (BM25 + vector), Reranking, Chunk size 500-1000 tokens.",
        ],
        "agent": [
            "AI Agents use LLMs to autonomously plan and execute tasks.",
            "Key patterns: ReAct (Reasoning + Acting), Plan-and-Execute, Reflection.",
            "Frameworks: LangGraph (state machines), CrewAI (multi-agent), AutoGen.",
        ],
        "default": [
            "This is a general topic. Key considerations include architecture, implementation, and evaluation.",
            "Modern approaches emphasize production readiness, scalability, and cost efficiency.",
        ]
    }
    
    topic = "default"
    for key in knowledge_base:
        if key in state.query.lower():
            topic = key
            break
    
    state.research_results = knowledge_base[topic]
    state.step_count += 1
    state.history.append(f"[{datetime.now().strftime('%H:%M:%S')}] Researcher: Found {len(state.research_results)} articles")
    state.status = "researched"
    
    print(f"  📄 Found {len(state.research_results)} relevant articles")
    return state


def writer_node(state: WorkflowState) -> WorkflowState:
    """Writer node: compose a draft from research results."""
    print(f"  ✍️  Writer: Composing draft...")
    time.sleep(0.3)
    
    # Simulated writing
    intro = f"# Analysis: {state.query}\n\n"
    body = "## Key Findings\n\n"
    for i, result in enumerate(state.research_results, 1):
        body += f"{i}. {result}\n"
    
    conclusion = f"\n## Conclusion\n\nBased on {len(state.research_results)} sources, "
    conclusion += f"this analysis covers the essential aspects of {state.query}. "
    conclusion += "Further research may be needed for specific implementation details."
    
    state.draft = intro + body + conclusion
    state.step_count += 1
    state.history.append(f"[{datetime.now().strftime('%H:%M:%S')}] Writer: Draft composed ({len(state.draft)} chars)")
    state.status = "drafted"
    
    print(f"  📝 Draft: {len(state.draft)} characters")
    return state


def reviewer_node(state: WorkflowState) -> WorkflowState:
    """Reviewer node: evaluate draft quality."""
    print(f"  🔎 Reviewer: Reviewing draft...")
    time.sleep(0.3)
    
    # Simulated review
    issues = []
    if len(state.draft) < 200:
        issues.append("Draft is too short. Need more detail.")
    if "Conclusion" not in state.draft:
        issues.append("Missing conclusion section.")
    if len(state.research_results) < 2:
        issues.append("Not enough research sources.")
    
    if issues:
        state.review_feedback = "NEEDS_REVISION: " + "; ".join(issues)
        state.status = "needs_revision"
    else:
        state.review_feedback = "APPROVED: Draft meets quality standards."
        state.status = "approved"
    
    state.step_count += 1
    state.history.append(f"[{datetime.now().strftime('%H:%M:%S')}] Reviewer: {state.status}")
    
    print(f"  {'✅' if state.status == 'approved' else '🔄'} Review: {state.review_feedback}")
    return state


def finalizer_node(state: WorkflowState) -> WorkflowState:
    """Finalizer node: prepare final output."""
    print(f"  🎯 Finalizer: Preparing output...")
    
    state.final_output = state.draft
    state.step_count += 1
    state.history.append(f"[{datetime.now().strftime('%H:%M:%S')}] Finalizer: Output ready")
    state.status = "completed"
    return state


# ============================================================
# Graph — connects nodes with edges
# ============================================================

class SimpleGraph:
    """Minimal state machine graph (simulates LangGraph)."""
    
    def __init__(self):
        self.nodes: dict[str, Callable] = {}
        self.edges: dict[str, str | Callable] = {}
        self.conditional_edges: dict[str, tuple[Callable, dict]] = {}
        self.entry_point: str | None = None
        self.checkpoints: list[dict] = []
    
    def add_node(self, name: str, func: Callable):
        self.nodes[name] = func
    
    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node
    
    def add_conditional_edge(self, from_node: str, condition: Callable, routes: dict):
        self.conditional_edges[from_node] = (condition, routes)
    
    def set_entry_point(self, name: str):
        self.entry_point = name
    
    def _checkpoint(self, state: WorkflowState, node_name: str):
        """Save state snapshot."""
        self.checkpoints.append({
            "node": node_name,
            "timestamp": datetime.now().isoformat(),
            "status": state.status,
            "step_count": state.step_count,
        })
    
    def run(self, initial_state: WorkflowState, max_iterations: int = 10) -> WorkflowState:
        """Execute the graph."""
        state = initial_state
        current = self.entry_point
        iteration = 0
        
        while current and current != "END" and iteration < max_iterations:
            iteration += 1
            print(f"\n--- Step {iteration}: [{current}] ---")
            
            # Execute node
            node_func = self.nodes.get(current)
            if node_func:
                state = node_func(state)
                self._checkpoint(state, current)
            
            # Route to next node
            if current in self.conditional_edges:
                condition, routes = self.conditional_edges[current]
                route_key = condition(state)
                current = routes.get(route_key, "END")
                print(f"  ➡️  Route: {route_key} → {current}")
            elif current in self.edges:
                current = self.edges[current]
            else:
                current = "END"
        
        return state


# ============================================================
# Build & Run Workflow
# ============================================================

def review_router(state: WorkflowState) -> str:
    """Route based on review result."""
    if state.status == "approved":
        return "approved"
    elif state.step_count > 6:
        return "approved"  # Force finish after too many iterations
    else:
        return "revise"


def main():
    print("=" * 60)
    print("📊 LangGraph-Style Workflow Demo")
    print("=" * 60)
    print("Simulates: State → Nodes → Conditional Edges → Checkpointing\n")
    
    # Build graph
    graph = SimpleGraph()
    
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("finalizer", finalizer_node)
    
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edge("reviewer", review_router, {
        "approved": "finalizer",
        "revise": "researcher",
    })
    graph.add_edge("finalizer", "END")
    
    # Run workflow
    queries = [
        "Explain RAG architecture and best practices",
        "How do AI agents work?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"🚀 Query: {query}")
        print(f"{'='*60}")
        
        initial_state = WorkflowState(query=query)
        final_state = graph.run(initial_state)
        
        print(f"\n📊 Results:")
        print(f"  Status: {final_state.status}")
        print(f"  Steps: {final_state.step_count}")
        print(f"  Output length: {len(final_state.final_output)} chars")
        print(f"  Review: {final_state.review_feedback}")
        
        print(f"\n📜 Execution History:")
        for entry in final_state.history:
            print(f"  {entry}")
        
        print(f"\n📄 Final Output (first 300 chars):")
        print(f"  {final_state.final_output[:300]}...")
    
    # Show checkpoints
    print(f"\n{'='*60}")
    print("💾 Checkpoints (last workflow):")
    for cp in graph.checkpoints[-6:]:
        print(f"  [{cp['node']}] status={cp['status']} steps={cp['step_count']}")
    
    print(f"\n{'='*60}")
    print("✅ Workflow demo completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
