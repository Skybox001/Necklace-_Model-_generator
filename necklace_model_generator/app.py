"""
FastAPI service: upload a necklace image -> get a photorealistic image of
an Indian model wearing it in a saree -> optionally apply natural-language
edits (e.g. "change the green stones to red") to the generated image.

Uses Pollinations.ai's `kontext` model, which requires input images to be
passed as PUBLIC URLs. This app therefore saves every uploaded/generated
image to its own /uploads or /generated endpoint first, builds a public
URL to it (using the incoming request's own host), and passes that URL to
Pollinations.

IMPORTANT: because of the public-URL requirement, image generation only
works once this app is deployed somewhere publicly reachable (e.g.
Render) — it will not work against http://127.0.0.1 during local
development, since Pollinations' servers can't reach your machine.

Run (after deploying):
    export POLLINATIONS_API_KEY=your_key_here   # https://auth.pollinations.ai
    uvicorn app:app --host 0.0.0.0 --port $PORT

Then open https://<your-deployed-domain>/ui/
"""

import os
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pollinations_client import generate_model_wearing_necklace, edit_image

app = FastAPI(title="Necklace-on-Model Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
GENERATED_DIR = os.path.join(BASE_DIR, "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


def _save_file(data: bytes, suffix: str = ".png") -> str:
    filename = f"{uuid.uuid4().hex}{suffix}"
    path = os.path.join(GENERATED_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)
    return filename


def _public_url(request: Request, filename: str) -> str:
    """Builds an absolute, publicly reachable URL to a saved file, using
    the incoming request's own scheme+host. On Render (or any real
    deployment) this is the actual public domain; on localhost this will
    produce a URL Pollinations can't reach — see module docstring."""
    base = str(request.base_url).rstrip("/")
    return f"{base}/generated/{filename}"


@app.get("/generated/{filename}")
def get_generated_image(filename: str):
    path = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@app.post("/generate")
async def generate(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    suffix = os.path.splitext(file.filename or "")[-1] or ".jpg"
    upload_filename = _save_file(image_bytes, suffix)
    upload_url = _public_url(request, upload_filename)

    try:
        result_bytes = generate_model_wearing_necklace(upload_url)
    except RuntimeError as e:
        raise HTTPException(502, str(e))

    result_filename = _save_file(result_bytes, ".png")
    return {"image_url": f"/generated/{result_filename}"}


@app.post("/edit")
async def edit(request: Request, instruction: str = Form(...), file: UploadFile = File(...)):
    """`file` is the previously generated image (re-uploaded by the
    frontend), `instruction` is the natural-language edit request."""
    image_bytes = await file.read()
    upload_filename = _save_file(image_bytes, ".png")
    upload_url = _public_url(request, upload_filename)

    try:
        result_bytes = edit_image(upload_url, instruction)
    except RuntimeError as e:
        raise HTTPException(502, str(e))

    result_filename = _save_file(result_bytes, ".png")
    return {"image_url": f"/generated/{result_filename}"}
