"""
Wraps Pollinations.ai's `kontext` image model for two operations this app
needs:

1. generate_model_wearing_necklace(image_url)
   Transform the necklace product photo (given as a public URL) into a
   photorealistic image of an Indian woman wearing a saree and that
   necklace.

2. edit_image(image_url, instruction)
   Apply a targeted natural-language edit to a previously generated image
   (given as a public URL) while preserving everything else.

Why Pollinations: after testing, Gemini's image-generation API requires a
billing-enabled project even on the free tier (see hf_client history in
git log), and Hugging Face's Inference Providers free allowance for
FLUX.1 Kontext Dev turned out to be a tiny $0.10/month credit routed
through a paid partner (fal-ai) — both exhausted almost immediately.
Pollinations.ai runs its own `kontext` model (FLUX Kontext under the
hood) on a separate, genuinely free credit system ("Pollen"), with no
credit card required.

IMPORTANT: Pollinations' API takes the input image as a **public URL**,
not a raw upload — the image must already be reachable on the internet
(this is why this app saves uploads to its own /generated/ or /uploads/
endpoint first, then passes that public URL to Pollinations). This means
image generation only works once this app itself is deployed somewhere
publicly reachable (e.g. Render) — it will not work against
http://127.0.0.1 during local development, since Pollinations' servers
can't reach your machine's localhost.

Get a free API key at https://enter.pollinations.ai/keys (no credit card).
Set it as POLLINATIONS_API_KEY in the environment.
"""

import os
import urllib.parse

import requests

BASE_URL = "https://gen.pollinations.ai/image"
MODEL_NAME = "kontext"


def _get_api_key() -> str:
    api_key = os.environ.get("POLLINATIONS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "POLLINATIONS_API_KEY is not set. Get a free key at "
            "https://enter.pollinations.ai/keys and set it as an "
            "environment variable."
        )
    return api_key


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


REQUEST_TIMEOUT = 300


def _call_pollinations(image_url: str, prompt: str) -> bytes:
    api_key = _get_api_key()
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"{BASE_URL}/{encoded_prompt}"
    params = {
        "model": MODEL_NAME,
        "image": image_url,
        "width": 1024,
        "height": 1024,
        "nologo": "true",
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Pollinations did not respond within {REQUEST_TIMEOUT} seconds "
            "— their generation queue may be overloaded. Please try again "
            "in a minute or two."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Could not reach Pollinations: {type(e).__name__}: {e}")
    if response.status_code != 200:
        detail = response.text[:500]
        if response.status_code == 402:
            detail = (
                "Your Pollinations pollen balance is empty (each image costs "
                "0.03 pollen). Earn free pollen via Quests or add pollen at "
                f"https://enter.pollinations.ai. Server said: {detail}"
            )
        raise RuntimeError(
            f"Pollinations API error ({response.status_code}): {detail}"
        )
    content_type = response.headers.get("content-type", "")
    if "image" not in content_type:
        raise RuntimeError(f"Pollinations did not return an image: {response.text[:500]}")
    return response.content


def generate_model_wearing_necklace(necklace_image_url: str) -> bytes:
    """Given a PUBLIC URL to a necklace product photo, returns image bytes
    of an Indian model wearing a saree and that necklace."""
    return _call_pollinations(necklace_image_url, GENERATE_PROMPT)


def edit_image(image_url: str, instruction: str) -> bytes:
    """Applies a natural-language edit instruction to an existing image
    (given as a PUBLIC URL), preserving everything not mentioned."""
    full_instruction = (
        f"{instruction.strip()}. "
        "Keep everything else in the image exactly the same: the model's "
        "face, expression, pose, hair, saree, background, and lighting must "
        "be unchanged. Only modify what was explicitly requested, and keep "
        "the rest of the necklace's design and metal color unchanged."
    )
    return _call_pollinations(image_url, full_instruction)
