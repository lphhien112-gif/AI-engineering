"""
💬 WebSocket Chat Demo — Real-time chat server
Chạy: python websocket_chat.py

Demonstrates WebSocket chat concepts with a simple TCP-based simulation.
For production: use fastapi + websockets library.
"""
import json
import time
import threading
import asyncio
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    id: str
    role: str  # user, assistant, system
    content: str
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        return {"id": self.id, "role": self.role, "content": self.content, "timestamp": self.timestamp}


class ChatRoom:
    """Manage chat conversations with streaming responses."""
    
    def __init__(self):
        self.conversations: dict[str, list[ChatMessage]] = {}
        self.msg_counter = 0
    
    def create_conversation(self, conv_id: str):
        self.conversations[conv_id] = []
        return conv_id
    
    def add_message(self, conv_id: str, role: str, content: str) -> ChatMessage:
        self.msg_counter += 1
        msg = ChatMessage(id=f"msg-{self.msg_counter:04d}", role=role, content=content)
        
        if conv_id not in self.conversations:
            self.create_conversation(conv_id)
        
        self.conversations[conv_id].append(msg)
        return msg
    
    def get_history(self, conv_id: str) -> list[dict]:
        return [msg.to_dict() for msg in self.conversations.get(conv_id, [])]
    
    def generate_response(self, conv_id: str, user_msg: str):
        """Generate streaming response (simulated AI)."""
        responses = {
            "hello": "Hello! I'm an AI assistant. How can I help you today?",
            "help": "I can help with coding, writing, analysis, and more. Just ask!",
            "transformer": "The Transformer architecture uses self-attention mechanisms to process sequences in parallel, replacing RNNs. Key components: Query-Key-Value attention, multi-head attention, positional encoding, and feed-forward layers.",
            "rag": "RAG (Retrieval Augmented Generation) combines search with LLMs. Pipeline: Question → Retrieve relevant chunks → Inject into prompt → Generate grounded answer. Key techniques: hybrid search, reranking, and evaluation with RAGAS.",
        }
        
        # Find best matching response
        lower_msg = user_msg.lower()
        response = "I understand your question. Let me think about that... In a production system, this would call an LLM API like GPT-4o or Claude to generate a contextual response based on the conversation history."
        
        for keyword, resp in responses.items():
            if keyword in lower_msg:
                response = resp
                break
        
        return response


class WebSocketSimulator:
    """Simulate WebSocket-like communication patterns."""
    
    def __init__(self):
        self.chat_room = ChatRoom()
        self.connections: dict[str, dict] = {}
    
    def connect(self, client_id: str):
        """Simulate WebSocket connection."""
        self.connections[client_id] = {
            "connected_at": datetime.now().isoformat(),
            "messages_sent": 0,
            "messages_received": 0,
        }
        
        conv_id = f"conv-{client_id}"
        self.chat_room.create_conversation(conv_id)
        
        print(f"  🔗 Client '{client_id}' connected")
        print(f"     WebSocket URL: ws://localhost:8765/ws/chat/{client_id}")
        return conv_id
    
    def disconnect(self, client_id: str):
        self.connections.pop(client_id, None)
        print(f"  🔌 Client '{client_id}' disconnected")
    
    def send_message(self, client_id: str, conv_id: str, text: str):
        """Simulate client sending a message."""
        # Record user message
        user_msg = self.chat_room.add_message(conv_id, "user", text)
        self.connections[client_id]["messages_sent"] += 1
        
        print(f"\n  👤 User: {text}")
        
        # Generate streaming response
        response = self.chat_room.generate_response(conv_id, text)
        
        # Simulate streaming (token by token)
        print(f"  🤖 Assistant: ", end="", flush=True)
        words = response.split()
        streamed = ""
        
        for i, word in enumerate(words):
            streamed += word + " "
            print(word, end=" ", flush=True)
            time.sleep(0.03)
        
        print()  # New line
        
        # Record assistant message
        self.chat_room.add_message(conv_id, "assistant", response)
        self.connections[client_id]["messages_received"] += 1
        
        return response


def demo_websocket_concepts():
    """Explain WebSocket concepts."""
    print("=== 1. WebSocket Concepts ===\n")
    
    print("  📋 WebSocket vs HTTP:")
    comparison = [
        ("Connection", "Persistent", "Per-request"),
        ("Direction", "Bidirectional", "Client → Server"),
        ("Overhead", "Low (frames)", "High (headers)"),
        ("Real-time", "Yes", "Polling needed"),
        ("Use case", "Chat, voice, collab", "REST APIs, pages"),
    ]
    
    print(f"  {'Feature':<14} {'WebSocket':<20} {'HTTP'}")
    print(f"  {'-'*50}")
    for feature, ws, http in comparison:
        print(f"  {feature:<14} {ws:<20} {http}")
    
    print(f"\n  📋 WebSocket Lifecycle:")
    print(f"    1. Client → HTTP Upgrade request")
    print(f"    2. Server → 101 Switching Protocols")
    print(f"    3. Bidirectional frames (text/binary)")
    print(f"    4. Either side can close")


def demo_connection_manager():
    """Demonstrate connection manager pattern."""
    print("\n=== 2. Connection Manager ===\n")
    
    print("  📋 ConnectionManager pattern (FastAPI):")
    print("""
    class ConnectionManager:
        active_connections: dict[str, WebSocket]
        
        async def connect(ws, client_id):
            await ws.accept()
            self.active[client_id] = ws
        
        def disconnect(client_id):
            self.active.pop(client_id)
        
        async def send(client_id, message):
            await self.active[client_id].send_json(message)
        
        async def broadcast(message):
            for ws in self.active.values():
                await ws.send_json(message)
    """)


def demo_chat_simulation():
    """Simulate a chat session."""
    print("=== 3. Chat Simulation ===\n")
    
    ws = WebSocketSimulator()
    
    # Connect
    conv_id = ws.connect("user-001")
    
    # Chat messages
    messages = [
        "Hello",
        "What is a transformer?",
        "Tell me about RAG",
        "Thanks for the help!",
    ]
    
    for msg in messages:
        ws.send_message("user-001", conv_id, msg)
        time.sleep(0.1)
    
    # Show history
    print(f"\n  📋 Conversation History:")
    for msg in ws.chat_room.get_history(conv_id):
        icon = "👤" if msg["role"] == "user" else "🤖"
        print(f"    {icon} [{msg['role']:<10}] {msg['content'][:80]}...")
    
    # Show connection stats
    print(f"\n  📊 Connection Stats:")
    stats = ws.connections.get("user-001", {})
    print(f"    Messages sent: {stats.get('messages_sent', 0)}")
    print(f"    Messages received: {stats.get('messages_received', 0)}")
    
    ws.disconnect("user-001")


def demo_message_types():
    """Show WebSocket message types for AI chat."""
    print("\n=== 4. Message Protocol ===\n")
    
    messages = [
        {"type": "user_message", "content": "Hello", "id": "msg-001"},
        {"type": "stream_start", "message_id": "msg-002"},
        {"type": "stream_token", "content": "Hello", "message_id": "msg-002"},
        {"type": "stream_token", "content": " there!", "message_id": "msg-002"},
        {"type": "stream_end", "message_id": "msg-002", "usage": {"tokens": 5}},
        {"type": "error", "code": "rate_limit", "message": "Too many requests"},
        {"type": "typing_indicator", "is_typing": True},
    ]
    
    print("  📋 WebSocket Message Types:")
    for msg in messages:
        print(f"    → {json.dumps(msg)}")


def main():
    print("=" * 60)
    print("💬 WebSocket Chat Demo")
    print("=" * 60)
    
    demo_websocket_concepts()
    demo_connection_manager()
    demo_chat_simulation()
    demo_message_types()
    
    print(f"\n{'='*60}")
    print("✅ WebSocket Chat Demo completed!")
    print("   For real WebSocket: pip install fastapi websockets uvicorn")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
