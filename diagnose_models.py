"""
Diagnostic: lists models available to your Gemini API key, and tries a
tiny generation call against a few known image-capable model names to see
which one actually works on your account's free tier.

Run:
    python diagnose_models.py
"""

import os
import sys

from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY is not set in this shell. Set it first, e.g.:")
    print('  $env:GEMINI_API_KEY="your_key_here"   (PowerShell)')
    sys.exit(1)

client = genai.Client(api_key=api_key)

print("=" * 70)
print("Listing all models visible to your API key...")
print("=" * 70)
try:
    for m in client.models.list():
        name = getattr(m, "name", "?")
        actions = getattr(m, "supported_actions", None)
        print(f"{name}  |  supported_actions={actions}")
except Exception as e:
    print(f"Could not list models: {e}")

print()
print("=" * 70)
print("Trying a minimal generation call against candidate image models...")
print("=" * 70)

candidates = [
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image",
    "nano-banana-pro-preview",
]

for model_name in candidates:
    print(f"\n--- Trying: {model_name} ---")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Generate a simple image of a red circle on a white background.",
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        parts = response.candidates[0].content.parts
        got_image = any(getattr(p, "inline_data", None) for p in parts)
        print(f"SUCCESS — got_image={got_image}")
    except Exception as e:
        msg = str(e)
        print(f"FAILED — {msg[:300]}")

print("\nDone. Use whichever model name printed SUCCESS in gemini_client.py's MODEL_NAME.")
