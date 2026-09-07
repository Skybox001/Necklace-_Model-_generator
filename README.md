# Necklace-on-Model Generator

Upload a necklace product photo → get a photorealistic image of an Indian
model wearing it with a saree → apply natural-language edits (e.g. "change
the green stones to red") to the generated image while keeping everything
else consistent.

## Model / API used

**Pollinations.ai's `kontext` model** (FLUX Kontext under the hood),
via their free HTTP image API. Free API key from
https://enter.pollinations.ai/keys, no credit card required.

### The path to get here (and why it matters for evaluation)

Three free-tier options were tried, in order:

1. **Gemini 2.5 Flash Image ("Nano Banana")** — the ideal fit
   architecturally (multi-image composition + precise local edits), but
   testing against a freshly created Google AI Studio key showed that
   **every current Gemini image-generation model**
   (`gemini-2.5-flash-image`, `gemini-3.1-flash-image`,
   `gemini-3.1-flash-image-preview`, `gemini-3.1-flash-lite-image`,
   `gemini-3-pro-image`, `nano-banana-pro-preview`) returns a hard
   `429 RESOURCE_EXHAUSTED` with `limit: 0` on the free API tier. Google's
   Developer API appears to now require a billing-enabled project for
   image generation, even though the free consumer AI Studio *website*
   still generates images by hand. Ruled out — not compatible with a
   genuinely free-tier requirement.

2. **FLUX.1 Kontext Dev via Hugging Face Inference Providers** —
   confirmed working (and produced a genuinely strong result — necklace
   design, gemstone colours, and metal tone were well preserved on a
   real test image). However, Hugging Face's free monthly allowance for
   this model turned out to be a flat **$0.10/month**, and this specific
   model is only routed through a paid partner (`fal-ai`) with no
   HF-hosted free alternative — so the free budget was exhausted after
   just 1–2 real generations, with the next free reset only on the 1st
   of the following month. Not viable for a live demo that needs to
   survive repeated evaluation.

3. **Pollinations.ai `kontext`** — a separate, independent free credit
   system ("Pollen") not tied to either of the above, no card required.
   This is what the deployed app actually uses.

**This history is included here deliberately**: it's the actual
problem-solving process for this assignment, not just the end result —
each "free tier" claim from a provider's marketing page had to be
verified against a real API call before trusting it, and two of three
fell over under real testing despite official documentation suggesting
they'd work.

### An important constraint this introduces

Pollinations' API takes the input image as a **public URL**, not a raw
file upload. This app therefore saves every uploaded/generated image to
its own `/generated/{filename}` endpoint first, builds a public URL to
it from the incoming request's own host, and passes *that* URL to
Pollinations.

**Practical effect: image generation only works once this app is
deployed somewhere publicly reachable (e.g. Render).** It will not work
against `http://127.0.0.1` during local development, since Pollinations'
servers cannot reach your machine's localhost. The `/ui` frontend and
error handling can still be checked locally; the actual generation call
needs the deployed URL.

## Tools / technologies

- Python 3, FastAPI + Uvicorn — backend API
- `requests` — calling Pollinations' HTTP image API
- Vanilla HTML/JS (`static/index.html`) — frontend, no build step
- Render — hosting (free web service tier)

## Prompting approach

**Step 1 — generation** (`pollinations_client.generate_model_wearing_necklace`):
The necklace product photo's public URL is passed to the `kontext` model
with a prompt instructing it to transform the scene into a photorealistic
studio portrait of an Indian woman in a saree wearing that exact
necklace — explicitly telling it to preserve the necklace's design,
metal colour, gemstone colours, and proportions rather than
reinterpreting them. This is a large scene transformation for an
image-editing model (going from an isolated product shot to a full
portrait), which is the main source of variability in results — see
Limitations.

**Step 2 — editing** (`pollinations_client.edit_image`):
Every edit call appends an explicit preservation clause ("keep
everything else exactly the same: face, pose, hair, saree, background,
lighting... only modify what was requested") alongside the user's
instruction — this is squarely within what Kontext-style models are
designed for (targeted local edits on an existing photo).

The frontend re-sends the *current* image on every edit call (rather
than the backend holding session state), so edits can be chained: change
stones to red, then change the metal to silver, etc.

## Project structure

```
necklace_model_generator/
├── app.py                    # FastAPI endpoints: /generate, /edit
├── pollinations_client.py    # Pollinations kontext wrapper + prompts
├── static/index.html         # Frontend (upload, generate, edit)
├── generated/                 # Uploaded + generated images saved here at runtime
├── reference/                 # Assignment's reference/test images (for demo)
├── requirements.txt
└── Procfile                   # Render start command
```

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key from https://enter.pollinations.ai/keys, then set it as an
environment variable (never commit it to source control):

```bash
export POLLINATIONS_API_KEY=your_key_here      # macOS/Linux
$env:POLLINATIONS_API_KEY="your_key_here"      # Windows PowerShell
```

Run locally to check the UI and server (generation itself needs a public
deployment — see above):

```bash
uvicorn app:app --reload --port 8000
```

**To actually test image generation**, deploy first (see below), then
open `https://<your-deployed-domain>/ui/`:
1. Upload a necklace image → click "Generate model image"
2. Once generated, use a preset button (e.g. "Stones → Red") or type a
   custom instruction, then click "Apply edit"

## API

- `POST /generate` (multipart, field `file`) → `{ "image_url": "/generated/xxx.png" }`
- `POST /edit` (multipart, fields `file` + `instruction`) → `{ "image_url": "/generated/xxx.png" }`

## Deploying (Render)

1. Push this repo to GitHub.
2. On render.com → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add an environment variable: `POLLINATIONS_API_KEY` = your key
   (Render → service → Environment tab — keeps the key server-side and
   out of the public repo/frontend).
6. Once deployed: `https://<your-app>.onrender.com/ui/`

## Limitations

- **Requires public deployment** to actually generate images (see
  above) — this is a direct consequence of Pollinations' URL-based API
  and the free-tier constraints described above, not a design choice.
- **Scene-transformation fidelity (step 1)**: asking an image-editing
  model to turn an isolated product photo into a full portrait of a
  person is a bigger transformation than Kontext-style models' typical
  use case (editing an existing photo's attributes). Necklace
  shape/colour is generally preserved reasonably well in testing, but
  very fine details (exact stone facet count, tiny engravings) may
  drift more than a purpose-built multi-image composition model would
  preserve.
- **Not deterministic**: results vary run to run, as with any diffusion
  model.
- **Edit drift**: multi-turn edits can occasionally cause small
  unintended changes elsewhere in the image despite the preservation
  clause in the prompt.
- **Free tier limits**: Pollinations' free "Pollen" allowance is finite;
  heavy testing in a short window could exhaust it. The app surfaces the
  API's error message rather than retrying automatically.
- **No face/identity consistency across separate `/generate` calls**:
  each new necklace upload generates a new model's face/scene from
  scratch.
