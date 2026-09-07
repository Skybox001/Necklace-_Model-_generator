"""
Wraps FLUX.1 Kontext Dev (black-forest-labs/FLUX.1-Kontext-dev) via
Hugging Face's Inference Providers for two operations this app needs:

1. generate_model_wearing_necklace(necklace_bytes)
   Transform the necklace product photo into a photorealistic image of an
   Indian woman wearing a saree and that necklace.

2. edit_image(image_bytes, instruction)
   Apply a targeted natural-language edit to a previously generated image
   (e.g. "change the green stones to red") while preserving everything
   else.

Why FLUX.1 Kontext Dev: it's an instruction-based image editing model
(single input image + text instruction -> new image) built specifically
to make targeted, localized edits while preserving the rest of the image
— which is exactly what the "change stone colour, keep everything else
identical" requirement needs. It's available on Hugging Face's free
Inference Providers tier (no credit card required).

Note: Gemini 2.5 Flash Image ("Nano Banana") was tried first since it's
purpose-built for this exact workflow, but as of testing, the Gemini
Developer API's free tier for every image-generation model
(gemini-2.5-flash-image, gemini-3.1-flash-image, gemini-3-pro-image,
nano-banana-pro-preview, etc.) returned a hard RESOURCE_EXHAUSTED / 0
free-quota error on a fresh AI Studio key — Google appears to now require
a billing-enabled project for image-generation API access even at their
lowest tier, unlike the free consumer AI Studio website. Since the
assignment requires a genuinely free-tier model, we switched to
Hugging Face + FLUX.1 Kontext Dev instead. See README for details.

Requires HF_TOKEN in the environment (a free personal access token from
https://huggingface.co/settings/tokens with "Inference Providers"
permission).
"""

import os
from io import BytesIO

from huggingface_hub import InferenceClient
from PIL import Image

MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"

_client = None


def get_client():
    global _client
    if _client is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set. Get a free token at "
                "https://huggingface.co/settings/tokens (with 'Inference "
                "Providers' permission) and set it as an environment "
                "variable."
            )
        _client = InferenceClient(api_key=token, provider="hf-inference")
    return _client


GENERATE_PROMPT = (
    "Transform this necklace product photo into a professional photorealistic "
    "studio portrait of a beautiful Indian woman with warm skin tone, "
    "photographed from the shoulders up, wearing an elegant traditional silk "
    "saree in soft neutral or gold tones, an elegant hairstyle with hair tied "
    "back, subtle traditional makeup, and a small bindi, in warm soft indoor "
    "studio lighting with a softly blurred elegant background. She is wearing "
    "exactly this necklace around her neck, resting naturally on her chest "
    "above the saree neckline. Preserve the necklace's exact design, metal "
    "color, every gemstone's color and cut, the pearl or bead drops, and "
    "overall proportions precisely as shown — do not invent new jewellery or "
    "change the stone colors, count, or shape. Photorealistic, premium "
    "jewellery photoshoot style, not an illustration."
)


def generate_model_wearing_necklace(necklace_image_bytes: bytes, mime_type: str = "image/jpeg") -> bytes:
    """Given a necklace product photo, returns PNG bytes of an Indian model
    wearing a saree and that necklace."""
    client = get_client()
    result_image = client.image_to_image(
        necklace_image_bytes,
        prompt=GENERATE_PROMPT,
        model=MODEL_NAME,
    )
    return _to_png_bytes(result_image)


def edit_image(image_bytes: bytes, instruction: str) -> bytes:
    """Applies a natural-language edit instruction to an existing image,
    preserving everything not mentioned in the instruction."""
    client = get_client()

    full_instruction = (
        f"{instruction.strip()}. "
        "Keep everything else in the image exactly the same: the model's "
        "face, expression, pose, hair, saree, background, and lighting must "
        "be unchanged. Only modify what was explicitly requested, and keep "
        "the rest of the necklace's design and metal color unchanged."
    )

    result_image = client.image_to_image(
        image_bytes,
        prompt=full_instruction,
        model=MODEL_NAME,
    )
    return _to_png_bytes(result_image)


def _to_png_bytes(pil_image: Image.Image) -> bytes:
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()
