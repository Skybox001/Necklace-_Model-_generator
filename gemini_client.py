"""
Wraps Google's Gemini 2.5 Flash Image model ("Nano Banana") for two
operations this app needs:

1. generate_model_wearing_necklace(necklace_bytes)
   Compose a photorealistic image of an Indian woman wearing a saree,
   wearing the exact necklace from the reference product photo.

2. edit_image(image_bytes, instruction)
   Apply a targeted natural-language edit to a previously generated
   image (e.g. "change the green stones to red") while preserving
   everything else (face, pose, saree, background, lighting, the rest
   of the jewellery).

Requires GEMINI_API_KEY in the environment (get a free key at
https://aistudio.google.com/apikey).
"""

import os
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image

MODEL_NAME = "gemini-2.5-flash-image"

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and set it as an "
                "environment variable."
            )
        _client = genai.Client(api_key=api_key)
    return _client


GENERATE_PROMPT = """You are compositing real jewellery onto a portrait photo.

Reference image: a product photo of a necklace.

Task: Generate a single photorealistic, professional studio portrait of a
beautiful Indian woman with a warm skin tone, wearing an elegant traditional
silk saree in soft neutral/gold tones, with an elegant hairstyle (hair tied
back or in a bun), subtle traditional makeup, and a soft bindi. She should be
photographed from the shoulders/chest up, facing slightly to one side,
in warm, soft indoor studio lighting with a softly blurred elegant
background (e.g. a warm interior with soft bokeh).

She must be wearing EXACTLY the necklace shown in the reference image around
her neck, resting naturally on her chest above the saree neckline. Preserve
the necklace's design, metal color (gold/rose gold/silver as shown), every
gemstone's color and cut, the pearl/bead drops, and overall proportions
EXACTLY as in the reference image. Do not invent new jewellery, do not
change the necklace's shape, stone colors, or stone count. Do not add
earrings, a nose ring, or other jewellery unless already implied.

The final image should look like a premium jewellery brand's real
photoshoot, not an illustration or CGI render."""


def generate_model_wearing_necklace(necklace_image_bytes: bytes, mime_type: str = "image/jpeg") -> bytes:
    """Given a necklace product photo, returns PNG bytes of an Indian model
    wearing a saree and that necklace."""
    client = get_client()
    necklace_image = Image.open(BytesIO(necklace_image_bytes)).convert("RGB")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[GENERATE_PROMPT, necklace_image],
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    return _extract_image_bytes(response)


def edit_image(image_bytes: bytes, instruction: str) -> bytes:
    """Applies a natural-language edit instruction to an existing image,
    preserving everything not mentioned in the instruction."""
    client = get_client()
    source_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    full_instruction = (
        f"Edit this image: {instruction.strip()}. "
        "Keep everything else in the image exactly the same: the model's "
        "face, expression, pose, hair, saree, background, and lighting must "
        "be unchanged. Only modify what was explicitly requested, and keep "
        "the rest of the necklace's design and metal color unchanged."
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[source_image, full_instruction],
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    return _extract_image_bytes(response)


def _extract_image_bytes(response) -> bytes:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise RuntimeError("Gemini returned no candidates. It may have refused the request.")

    parts = candidates[0].content.parts
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data is not None:
            return inline_data.data

    # No image part found — surface any text response (often a refusal or
    # safety message) so the caller can show something useful.
    text_parts = [p.text for p in parts if getattr(p, "text", None)]
    message = " ".join(text_parts) if text_parts else "Gemini returned no image."
    raise RuntimeError(f"No image returned by Gemini: {message}")
