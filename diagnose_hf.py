"""
Diagnostic: verifies your HF_TOKEN works and can actually call FLUX.1
Kontext Dev for image-to-image generation.

Run:
    python diagnose_hf.py
"""

import os
import sys
from io import BytesIO

from huggingface_hub import InferenceClient
from PIL import Image

token = os.environ.get("HF_TOKEN")
if not token:
    print("HF_TOKEN is not set in this shell. Set it first, e.g.:")
    print('  $env:HF_TOKEN="your_token_here"   (PowerShell)')
    sys.exit(1)

client = InferenceClient(api_key=token)

print("Creating a tiny test image and sending it to FLUX.1 Kontext Dev...")
test_img = Image.new("RGB", (256, 256), color=(200, 50, 50))
buf = BytesIO()
test_img.save(buf, format="PNG")
image_bytes = buf.getvalue()

try:
    result = client.image_to_image(
        image_bytes,
        prompt="Change the background color to blue.",
        model="black-forest-labs/FLUX.1-Kontext-dev",
    )
    result.save("diagnose_hf_output.png")
    print("SUCCESS — saved result to diagnose_hf_output.png. Open it to check.")
except Exception as e:
    print(f"FAILED — {e}")
