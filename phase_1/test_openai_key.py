"""
Minimal test: load .env and make one OpenAI API call.
Run from phase_1: python test_openai_key.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from starter/ (same as evaluation_agent.py)
script_dir = Path(__file__).resolve().parent
project_starter_root = script_dir.parent
env_path = project_starter_root / ".env"
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("FAIL: OPENAI_API_KEY not found in .env")
    print(f"  Checked: {env_path}")
    exit(1)

print(f"Loaded key from: {env_path}")
print(f"Key starts with: {api_key[:15]}...")
print("Calling OpenAI API (one chat completion)...")

try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5,
    )
    text = resp.choices[0].message.content
    print(f"SUCCESS: API responded with: {text!r}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    if hasattr(e, "response") and e.response is not None:
        print(f"  Status: {e.response.status_code}")
    exit(1)
