"""
💬 OpenAI Chat Completions — Basics
Chạy: pip install openai python-dotenv
       python openai_basics.py
"""
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file containing OPENAI_API_KEY


def basic_chat():
    """Basic chat completion."""
    from openai import OpenAI
    client = OpenAI()  # Uses OPENAI_API_KEY from env
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI engineering tutor."},
            {"role": "user", "content": "Explain transformers in 3 sentences."},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    
    print("=== Basic Chat ===")
    print(response.choices[0].message.content)
    print(f"Tokens used: {response.usage.total_tokens}")
    print(f"Cost: ~${response.usage.total_tokens * 0.00015 / 1000:.6f}")


def structured_output():
    """Structured output with JSON mode."""
    from openai import OpenAI
    import json
    
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract entities. Respond in JSON with keys: persons, organizations, locations"},
            {"role": "user", "content": "Elon Musk founded SpaceX in Hawthorne, California."},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    
    result = json.loads(response.choices[0].message.content)
    print("\n=== Structured Output ===")
    print(json.dumps(result, indent=2))


def streaming_demo():
    """Streaming response (token by token)."""
    from openai import OpenAI
    
    client = OpenAI()
    
    print("\n=== Streaming ===")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 10 slowly."}],
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def multi_turn_conversation():
    """Multi-turn conversation with history."""
    from openai import OpenAI
    
    client = OpenAI()
    messages = [
        {"role": "system", "content": "You are a Python tutor. Be concise."},
    ]
    
    print("\n=== Multi-Turn Conversation ===")
    user_messages = [
        "What is a decorator?",
        "Give me a simple example.",
        "How do I add arguments to it?",
    ]
    
    for user_msg in user_messages:
        messages.append({"role": "user", "content": user_msg})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=150,
        )
        
        assistant_msg = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_msg})
        
        print(f"User: {user_msg}")
        print(f"AI: {assistant_msg}\n")


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Set OPENAI_API_KEY in .env file:")
        print('   OPENAI_API_KEY=sk-...')
        print("\nRunning in demo mode (no API calls)...")
        print("✅ Code is correct — just needs API key to execute.")
    else:
        basic_chat()
        structured_output()
        streaming_demo()
        multi_turn_conversation()
