"""
🔧 MCP Tool Server Demo — Simulated MCP server pattern
Chạy: python mcp_tool_server.py

Demonstrates MCP concepts: tool registration, schema validation,
tool execution, resource serving — no external deps needed.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ============================================================
# MCP Types (simplified)
# ============================================================

@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Any = None

@dataclass
class Resource:
    uri: str
    name: str
    description: str
    content: str
    mime_type: str = "application/json"

@dataclass
class ToolResult:
    content: str
    is_error: bool = False


# ============================================================
# MCP Server
# ============================================================

class MCPServer:
    """Simplified MCP server demonstrating the protocol pattern."""
    
    def __init__(self, name: str):
        self.name = name
        self.tools: dict[str, Tool] = {}
        self.resources: dict[str, Resource] = {}
        self._log: list[str] = []
    
    def register_tool(self, tool: Tool):
        """Register a tool with the server."""
        self.tools[tool.name] = tool
        self._log.append(f"Registered tool: {tool.name}")
    
    def register_resource(self, resource: Resource):
        """Register a resource with the server."""
        self.resources[resource.uri] = resource
        self._log.append(f"Registered resource: {resource.uri}")
    
    # --- Protocol Methods ---
    
    def handle_list_tools(self) -> list[dict]:
        """MCP: tools/list → Return available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self.tools.values()
        ]
    
    def handle_call_tool(self, name: str, arguments: dict) -> ToolResult:
        """MCP: tools/call → Execute a tool."""
        tool = self.tools.get(name)
        if not tool:
            return ToolResult(content=f"Unknown tool: {name}", is_error=True)
        
        # Validate required params
        required = tool.input_schema.get("required", [])
        missing = [r for r in required if r not in arguments]
        if missing:
            return ToolResult(
                content=f"Missing required parameters: {missing}",
                is_error=True,
            )
        
        # Execute
        try:
            result = tool.handler(**arguments)
            self._log.append(f"Called tool: {name}({arguments}) → success")
            return ToolResult(content=json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            self._log.append(f"Called tool: {name}({arguments}) → error: {e}")
            return ToolResult(content=f"Error: {str(e)}", is_error=True)
    
    def handle_list_resources(self) -> list[dict]:
        """MCP: resources/list → Return available resources."""
        return [
            {
                "uri": res.uri,
                "name": res.name,
                "description": res.description,
                "mimeType": res.mime_type,
            }
            for res in self.resources.values()
        ]
    
    def handle_read_resource(self, uri: str) -> str:
        """MCP: resources/read → Return resource content."""
        resource = self.resources.get(uri)
        if not resource:
            raise ValueError(f"Unknown resource: {uri}")
        return resource.content


# ============================================================
# Sample Tools
# ============================================================

def search_products(query: str, category: str = None, limit: int = 5) -> dict:
    """Search products in the catalog by keyword."""
    products = [
        {"id": 1, "name": "Wireless Headphones", "category": "electronics", "price": 79.99, "rating": 4.5},
        {"id": 2, "name": "Python Cookbook", "category": "books", "price": 45.00, "rating": 4.8},
        {"id": 3, "name": "Mechanical Keyboard", "category": "electronics", "price": 129.99, "rating": 4.7},
        {"id": 4, "name": "AI Engineering Guide", "category": "books", "price": 55.00, "rating": 4.9},
        {"id": 5, "name": "USB-C Hub", "category": "electronics", "price": 35.99, "rating": 4.3},
        {"id": 6, "name": "Standing Desk", "category": "furniture", "price": 399.00, "rating": 4.6},
        {"id": 7, "name": "Monitor Light Bar", "category": "electronics", "price": 49.99, "rating": 4.4},
    ]
    
    results = products
    if category:
        results = [p for p in results if p["category"] == category]
    if query:
        results = [p for p in results if query.lower() in p["name"].lower() or query.lower() in p["category"]]
    
    return {"products": results[:limit], "total": len(results), "query": query}


def get_user_profile(user_id: str) -> dict:
    """Get user profile by ID."""
    users = {
        "user-001": {"name": "Nguyen Van A", "email": "a@example.com", "plan": "pro", "joined": "2024-01-15"},
        "user-002": {"name": "Tran Thi B", "email": "b@example.com", "plan": "free", "joined": "2024-06-20"},
    }
    user = users.get(user_id)
    if not user:
        raise ValueError(f"User not found: {user_id}")
    return {**user, "user_id": user_id}


def create_ticket(title: str, priority: str = "medium", description: str = "") -> dict:
    """Create a support ticket."""
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return {
        "ticket_id": ticket_id,
        "title": title,
        "priority": priority,
        "description": description,
        "status": "open",
        "created_at": datetime.now().isoformat(),
    }


# ============================================================
# Demo
# ============================================================

def main():
    print("=" * 60)
    print("🔧 MCP Tool Server Demo")
    print("=" * 60)
    print("Demonstrates: Tool registration, Schema validation, Execution\n")
    
    # 1. Create server
    server = MCPServer("demo-server")
    
    # 2. Register tools
    server.register_tool(Tool(
        name="search_products",
        description="Search products in the catalog by keyword, with optional category filter",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords"},
                "category": {"type": "string", "enum": ["electronics", "books", "furniture"]},
                "limit": {"type": "integer", "default": 5, "description": "Max results (1-20)"},
            },
            "required": ["query"],
        },
        handler=search_products,
    ))
    
    server.register_tool(Tool(
        name="get_user_profile",
        description="Look up a user's profile information by their user ID",
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User ID (format: user-XXX)"},
            },
            "required": ["user_id"],
        },
        handler=get_user_profile,
    ))
    
    server.register_tool(Tool(
        name="create_ticket",
        description="Create a new support ticket for a customer issue",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Ticket title"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "description": {"type": "string", "description": "Detailed description"},
            },
            "required": ["title"],
        },
        handler=create_ticket,
    ))
    
    # 3. Register resources
    server.register_resource(Resource(
        uri="config://database-schema",
        name="Database Schema",
        description="Current database table definitions",
        content=json.dumps({
            "tables": ["users", "products", "orders", "tickets"],
            "version": "2.1.0",
        }),
    ))
    
    # 4. Simulate MCP protocol calls
    print("📋 1. List Tools (tools/list)")
    print("-" * 40)
    tools = server.handle_list_tools()
    for t in tools:
        params = list(t["inputSchema"].get("properties", {}).keys())
        print(f"  🔧 {t['name']}({', '.join(params)})")
        print(f"     {t['description']}")
    
    print(f"\n📋 2. List Resources (resources/list)")
    print("-" * 40)
    resources = server.handle_list_resources()
    for r in resources:
        print(f"  📄 {r['uri']} — {r['name']}")
    
    print(f"\n📋 3. Call Tools (tools/call)")
    print("-" * 40)
    
    # Test calls
    test_calls = [
        ("search_products", {"query": "keyboard", "category": "electronics"}),
        ("search_products", {"query": "book"}),
        ("get_user_profile", {"user_id": "user-001"}),
        ("get_user_profile", {"user_id": "user-999"}),  # Error case
        ("create_ticket", {"title": "Login issue", "priority": "high", "description": "Cannot login since update"}),
        ("search_products", {}),  # Missing required param
    ]
    
    for tool_name, args in test_calls:
        print(f"\n  → {tool_name}({json.dumps(args)})")
        result = server.handle_call_tool(tool_name, args)
        if result.is_error:
            print(f"  ❌ Error: {result.content}")
        else:
            # Show truncated result
            content = result.content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"  ✅ {content}")
    
    print(f"\n📋 4. Read Resource (resources/read)")
    print("-" * 40)
    content = server.handle_read_resource("config://database-schema")
    print(f"  📄 {content}")
    
    print(f"\n📋 5. Server Logs")
    print("-" * 40)
    for log in server._log:
        print(f"  {log}")
    
    print(f"\n{'='*60}")
    print("✅ MCP Tool Server Demo completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
