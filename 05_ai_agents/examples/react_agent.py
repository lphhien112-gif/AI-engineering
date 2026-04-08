"""
🤖 ReAct Agent Demo — Tool-using agent chạy local (không cần API key)
Chạy: python react_agent.py
"""
import json
import re
import math
import random
from datetime import datetime


# ============================================================
# Tools — agent sẽ dùng những tools này
# ============================================================

def calculator(expression: str) -> dict:
    """Evaluate a mathematical expression safely. Supports +, -, *, /, **, sqrt, sin, cos, log."""
    try:
        allowed = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "abs": abs, "round": round,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"expression": expression, "result": round(result, 6)}
    except Exception as e:
        return {"error": str(e)}


def get_weather(city: str) -> dict:
    """Get current weather for a city. Returns temperature, humidity, and conditions."""
    weather_db = {
        "hanoi": {"temp": 28, "humidity": 85, "condition": "thunderstorm", "rain_chance": 90},
        "hcmc": {"temp": 32, "humidity": 75, "condition": "sunny", "rain_chance": 20},
        "tokyo": {"temp": 18, "humidity": 60, "condition": "cloudy", "rain_chance": 40},
        "london": {"temp": 12, "humidity": 80, "condition": "rainy", "rain_chance": 85},
        "new york": {"temp": 22, "humidity": 55, "condition": "clear", "rain_chance": 10},
    }
    data = weather_db.get(city.lower(), {"temp": 25, "humidity": 65, "condition": "unknown"})
    data["city"] = city
    data["unit"] = "celsius"
    return data


def search_knowledge(query: str) -> dict:
    """Search a knowledge base for relevant information. Returns matching articles."""
    kb = {
        "python": "Python is a high-level programming language. It supports OOP, functional programming, and async/await. Popular for AI/ML with libraries like PyTorch, scikit-learn.",
        "machine learning": "Machine Learning is building systems that learn from data. Types: supervised (classification, regression), unsupervised (clustering), reinforcement learning.",
        "transformer": "Transformer architecture uses self-attention mechanism. Introduced in 'Attention is All You Need' (2017). Foundation of GPT, BERT, Claude.",
        "rag": "RAG (Retrieval-Augmented Generation) combines retrieval with generation. Steps: embed query → search vector DB → retrieve chunks → generate with context.",
        "docker": "Docker packages applications in containers. Key commands: build, run, push. Use multi-stage builds for smaller images. Docker Compose for multi-service apps.",
    }
    results = []
    for key, value in kb.items():
        if any(q_word in key for q_word in query.lower().split()):
            results.append({"topic": key, "content": value})
    
    if not results:
        results.append({"topic": "general", "content": f"No specific results for '{query}'. Try a different query."})
    return {"query": query, "results": results, "count": len(results)}


def get_time(timezone: str = "UTC+7") -> dict:
    """Get current date and time."""
    now = datetime.now()
    return {"datetime": now.strftime("%Y-%m-%d %H:%M:%S"), "timezone": timezone, "day": now.strftime("%A")}


# ============================================================
# ReAct Agent
# ============================================================

TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "search_knowledge": search_knowledge,
    "get_time": get_time,
}


class SimpleReActAgent:
    """ReAct agent that reasons step-by-step and uses tools."""
    
    def __init__(self, tools: dict, max_steps: int = 5):
        self.tools = tools
        self.max_steps = max_steps
        self.trace: list[dict] = []
    
    def _format_tools(self) -> str:
        lines = []
        for name, func in self.tools.items():
            doc = func.__doc__ or "No description"
            lines.append(f"  - {name}: {doc.strip()}")
        return "\n".join(lines)
    
    def _simple_reasoning(self, question: str, history: list[dict]) -> tuple[str, str, dict]:
        """Simulate LLM reasoning (rule-based for demo without API key)."""
        q_lower = question.lower()
        
        # Check if we already have observations
        observations = [h for h in history if h["type"] == "observation"]
        
        if not observations:
            # First step — decide what to do
            if any(word in q_lower for word in ["weather", "temperature", "rain", "umbrella"]):
                # Extract city
                cities = ["hanoi", "hcmc", "tokyo", "london", "new york"]
                city = next((c for c in cities if c in q_lower), "hanoi")
                return (
                    f"I need to check the weather to answer this question. Let me look up weather for {city}.",
                    "get_weather", {"city": city}
                )
            elif any(word in q_lower for word in ["calculate", "math", "compute", "+", "-", "*", "/"]):
                expr = re.findall(r'[\d\.\+\-\*\/\(\)\s\^]+', question)
                expression = expr[0].strip() if expr else "1+1"
                expression = expression.replace("^", "**")
                return (
                    f"I need to calculate: {expression}",
                    "calculator", {"expression": expression}
                )
            elif any(word in q_lower for word in ["time", "date", "today", "day"]):
                return (
                    "Let me check the current time.",
                    "get_time", {"timezone": "UTC+7"}
                )
            else:
                query = " ".join(question.split()[:3])
                return (
                    f"I need to search for information about this topic.",
                    "search_knowledge", {"query": query}
                )
        else:
            # We have observations — generate final answer
            last_obs = observations[-1]["data"]
            
            if "temp" in str(last_obs):
                data = last_obs
                rain = data.get("rain_chance", 0)
                umbrella = "Yes, bring an umbrella! ☂️" if rain > 50 else "No umbrella needed. ☀️"
                answer = (
                    f"Weather in {data.get('city', 'N/A')}: {data.get('temp')}°C, "
                    f"{data.get('condition')}, humidity {data.get('humidity')}%, "
                    f"rain chance {rain}%. {umbrella}"
                )
            elif "result" in str(last_obs):
                answer = f"The result is: {last_obs.get('result', last_obs)}"
            elif "datetime" in str(last_obs):
                answer = f"Current time: {last_obs['datetime']} ({last_obs['timezone']}), {last_obs['day']}"
            else:
                results = last_obs.get("results", [])
                if results:
                    answer = " | ".join(r.get("content", "") for r in results[:2])
                else:
                    answer = f"Based on my research: {json.dumps(last_obs)}"
            
            return (f"I now have enough information to answer.", None, {"answer": answer})
    
    def run(self, question: str) -> str:
        """Run agent on a question."""
        self.trace = []
        history = []
        
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}")
        
        for step in range(self.max_steps):
            thought, action, params = self._simple_reasoning(question, history)
            
            print(f"\n💭 Thought {step+1}: {thought}")
            
            if action is None:
                # Final answer
                answer = params.get("answer", "I couldn't determine an answer.")
                print(f"\n✅ Final Answer: {answer}")
                self.trace.append({"type": "answer", "content": answer})
                return answer
            
            # Execute tool
            print(f"🔧 Action: {action}({json.dumps(params)})")
            
            if action in self.tools:
                result = self.tools[action](**params)
                print(f"👁️ Observation: {json.dumps(result, indent=2)}")
                
                self.trace.append({
                    "type": "step",
                    "thought": thought,
                    "action": action,
                    "params": params,
                    "observation": result,
                })
                history.append({"type": "observation", "data": result})
            else:
                print(f"❌ Unknown tool: {action}")
                history.append({"type": "error", "data": f"Unknown tool: {action}"})
        
        return "Reached maximum steps without finding an answer."


def main():
    print("🤖 ReAct Agent Demo")
    print("=" * 60)
    print("This agent reasons step-by-step and uses tools to answer questions.")
    print("No API key needed — uses rule-based reasoning for demonstration.\n")
    
    agent = SimpleReActAgent(TOOLS, max_steps=5)
    
    # Test queries
    questions = [
        "What's the weather in Hanoi? Should I bring an umbrella?",
        "Calculate 25 * 4 + 100 / 5",
        "What time is it today?",
        "Tell me about machine learning",
        "What is a transformer architecture?",
    ]
    
    for q in questions:
        answer = agent.run(q)
    
    # Show agent trace for last query
    print(f"\n{'='*60}")
    print("📋 Agent Trace (last query):")
    print(f"{'='*60}")
    for i, step in enumerate(agent.trace):
        print(f"  Step {i+1}: {step['type']}")
        if step["type"] == "step":
            print(f"    Tool: {step['action']}")
            print(f"    Result: {json.dumps(step['observation'])[:100]}...")
    
    print(f"\n{'='*60}")
    print("✅ ReAct Agent Demo completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
