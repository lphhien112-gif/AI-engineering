"""
🚀 FastAPI Server Demo — API design with streaming
Chạy: pip install fastapi uvicorn
       python fastapi_server.py
       # Open http://localhost:8000/docs for Swagger UI

NOTE: Runs without external dependencies for demo. Install fastapi+uvicorn for real use.
"""
import json
import time
import asyncio
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading


class SimpleAPIServer(BaseHTTPRequestHandler):
    """Simplified API server demonstrating FastAPI-like patterns."""
    
    routes = {}
    
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == "/health":
            self.json_response(200, {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": True,
                "version": "1.0.0",
            })
        
        elif path == "/docs":
            self.json_response(200, {
                "openapi": "3.0.0",
                "info": {"title": "AI Model API", "version": "1.0.0"},
                "paths": {
                    "/health": {"get": {"summary": "Health check"}},
                    "/predict": {"post": {"summary": "Model prediction"}},
                    "/chat/stream": {"post": {"summary": "Streaming chat"}},
                },
            })
        
        elif path == "/metrics":
            self.json_response(200, {
                "total_requests": RequestMetrics.total,
                "avg_latency_ms": RequestMetrics.avg_latency(),
                "uptime_seconds": time.time() - RequestMetrics.start_time,
            })
        
        else:
            self.json_response(404, {"error": "Not found"})
    
    def do_POST(self):
        path = urlparse(self.path).path
        
        # Read body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length else '{}'
        
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.json_response(400, {"error": "Invalid JSON"})
            return
        
        start = time.time()
        
        if path == "/predict":
            self.handle_predict(data)
        elif path == "/chat":
            self.handle_chat(data)
        elif path == "/chat/stream":
            self.handle_stream(data)
        else:
            self.json_response(404, {"error": "Not found"})
        
        RequestMetrics.record(time.time() - start)
    
    def handle_predict(self, data):
        """Standard prediction endpoint."""
        # Validate
        text = data.get("text", "")
        if not text:
            self.json_response(400, {"error": "Field 'text' is required"})
            return
        
        if len(text) > 10000:
            self.json_response(400, {"error": "Text too long (max 10000 chars)"})
            return
        
        model = data.get("model", "gpt-4o-mini")
        temperature = data.get("temperature", 0.7)
        
        # Simulate inference
        time.sleep(0.1)
        word_count = len(text.split())
        
        self.json_response(200, {
            "text": f"Processed {word_count} words with {model}",
            "model": model,
            "usage": {
                "prompt_tokens": word_count,
                "completion_tokens": 20,
                "total_tokens": word_count + 20,
            },
            "latency_ms": 100,
        })
    
    def handle_chat(self, data):
        """Chat completion endpoint."""
        messages = data.get("messages", [])
        
        # Validate
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                self.json_response(400, {"error": "Each message needs 'role' and 'content'"})
                return
            if msg["role"] not in ("system", "user", "assistant"):
                self.json_response(400, {"error": f"Invalid role: {msg['role']}"})
                return
        
        # Simulate response
        last_msg = messages[-1]["content"] if messages else ""
        
        self.json_response(200, {
            "message": {
                "role": "assistant",
                "content": f"I received your message: '{last_msg[:50]}...' (simulated response)",
            },
            "model": data.get("model", "gpt-4o-mini"),
            "usage": {"total_tokens": 50},
        })
    
    def handle_stream(self, data):
        """SSE streaming endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        
        # Simulate streaming tokens
        words = "The Transformer architecture uses self-attention to process sequences in parallel, enabling much faster training than RNNs.".split()
        
        for word in words:
            chunk = json.dumps({
                "choices": [{"delta": {"content": word + " "}}],
            })
            self.wfile.write(f"data: {chunk}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)
        
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
    
    def json_response(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def log_message(self, format, *args):
        pass  # Suppress default logging


class RequestMetrics:
    total = 0
    latencies = []
    start_time = time.time()
    
    @classmethod
    def record(cls, latency):
        cls.total += 1
        cls.latencies.append(latency * 1000)
    
    @classmethod
    def avg_latency(cls):
        return round(sum(cls.latencies) / len(cls.latencies), 1) if cls.latencies else 0


def demo_api_concepts():
    """Demonstrate API design concepts."""
    print("=== API Design Concepts ===\n")
    
    print("📋 RESTful Endpoints:")
    endpoints = [
        ("GET", "/health", "Health check", "200: {status, model_loaded}"),
        ("GET", "/docs", "API documentation", "200: OpenAPI schema"),
        ("GET", "/metrics", "Performance metrics", "200: {total_requests, avg_latency}"),
        ("POST", "/predict", "Model prediction", "200: {text, usage, latency_ms}"),
        ("POST", "/chat", "Chat completion", "200: {message, model, usage}"),
        ("POST", "/chat/stream", "Streaming chat (SSE)", "200: text/event-stream"),
    ]
    
    print(f"  {'Method':<7} {'Path':<18} {'Description':<25} {'Response'}")
    print(f"  {'-'*75}")
    for method, path, desc, response in endpoints:
        print(f"  {method:<7} {path:<18} {desc:<25} {response}")
    
    print(f"\n📋 HTTP Status Codes for AI APIs:")
    codes = [
        ("200", "Success", "Normal response"),
        ("400", "Bad Request", "Invalid input (missing field, too long)"),
        ("401", "Unauthorized", "Invalid/missing API key or JWT"),
        ("429", "Rate Limited", "Too many requests"),
        ("500", "Server Error", "Model crash, OOM"),
        ("503", "Service Unavailable", "Model loading, cold start"),
    ]
    
    for code, name, desc in codes:
        print(f"  {code} {name:<22} {desc}")
    
    print(f"\n📋 Request Validation (Pydantic-style):")
    print("""
    class PredictionRequest:
        text: str           # Required, min_length=1, max_length=10000
        model: str          # Default: "gpt-4o-mini"
        temperature: float  # Default: 0.7, range: [0.0, 2.0]
        max_tokens: int     # Default: 1000, range: [1, 4096]
    """)


def demo_middleware():
    """Demonstrate middleware concepts."""
    print("\n=== Middleware Patterns ===\n")
    
    middlewares = [
        ("Timing", "Log request duration", "X-Process-Time-Ms header"),
        ("CORS", "Cross-origin requests", "Allow specific origins"),
        ("Auth", "Verify JWT/API key", "401 if invalid"),
        ("Rate Limit", "Throttle requests", "429 if exceeded"),
        ("Logging", "Request/response logging", "Structured JSON logs"),
        ("Error Handler", "Catch unhandled errors", "500 with details"),
    ]
    
    print(f"  {'Middleware':<15} {'Purpose':<28} {'Output'}")
    print(f"  {'-'*60}")
    for name, purpose, output in middlewares:
        print(f"  {name:<15} {purpose:<28} {output}")
    
    print(f"\n  Request flow: Client → CORS → Auth → Rate Limit → Handler → Timing → Response")


def demo_streaming():
    """Demonstrate streaming concepts."""
    print("\n=== Streaming (SSE) ===\n")
    
    print("  📋 SSE Format:")
    print("    data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\\n\\n")
    print("    data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\\n\\n")
    print("    data: [DONE]\\n\\n")
    
    print(f"\n  📋 Simulating token stream:")
    words = "FastAPI makes building AI APIs simple and fast.".split()
    
    full_text = ""
    for i, word in enumerate(words):
        full_text += word + " "
        print(f"    Token {i+1:2d}: \"{word}\" → \"{full_text.strip()}\"")
        time.sleep(0.03)
    
    print(f"\n  ✅ Stream complete: {len(words)} tokens")


def main():
    print("=" * 60)
    print("🚀 FastAPI Server Demo")
    print("=" * 60)
    
    demo_api_concepts()
    demo_middleware()
    demo_streaming()
    
    # Start server
    print(f"\n{'='*60}")
    print("🌐 Starting API server...")
    print("=" * 60)
    
    port = 8000
    server = HTTPServer(("0.0.0.0", port), SimpleAPIServer)
    
    print(f"  🚀 Server running at http://localhost:{port}")
    print(f"  📋 Endpoints:")
    print(f"     GET  http://localhost:{port}/health")
    print(f"     GET  http://localhost:{port}/docs")
    print(f"     GET  http://localhost:{port}/metrics")
    print(f"     POST http://localhost:{port}/predict")
    print(f"     POST http://localhost:{port}/chat")
    print(f"     POST http://localhost:{port}/chat/stream")
    print(f"\n  Press Ctrl+C to stop.\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  ⏹️  Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
