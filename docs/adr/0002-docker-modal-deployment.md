# 0002 — Docker + Modal-via-Dockerfile for cloud deploy

**Status**: accepted

## Context

`scripts/main_modal.py` builds the Modal image via `modal.Image.debian_slim().pip_install_from_requirements(...)` and `add_local_python_source(...)`. Modal's CLI walks every import in the source to build a list of package paths, then calls `os.path.commonpath` on them to find the function's base directories. On Windows hosts where the project lives on a different drive letter from the Python install (e.g. project on `E:`, conda on `C:`), `commonpath` raises `ValueError: Paths don't have the same drive` and the deploy fails. Pinning `modal<0.65` works but freezes the SDK; switching to WSL or moving the project are both higher-cost fixes than the bug warrants.

## Decision

Cloud deploys use a two-file Docker setup plus a thin Modal wrapper. The FastAPI app moves to a `modal`-import-free script so the same object is shipped by both `docker run` and `modal deploy`.

- **`Dockerfile`** (CPU, `python:3.11-slim`): for local dev and CI repro. Small, fast, no CUDA.
- **`Dockerfile.cuda`** (GPU, `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` + Python 3.11): for Modal deploy, matches the existing T4 production target.
- **`scripts/main_docker.py`**: module-level `app = FastAPI(...)` with all routes from `scripts/main_modal.py` minus the `modal.*` decoration. Imports `modal` only to call `Volume.from_name("hf_embedding_cache").commit()` after ingest — wrapped in `try/except` so plain `docker run` (no Modal token) is a silent no-op.
- **`scripts/main_modal_docker.py`**: pure Modal plumbing. Picks the image source from the `IMAGE_TAG` env var — set: `modal.Image.from_registry(IMAGE_TAG)` (CI / registry mode), unset: `modal.Image.from_dockerfile("./Dockerfile.cuda")` (local dev / Modal builder). Imports `from scripts.main_docker import app as fastapi_app` to wrap with `@modal.asgi_app`.
- **HF embedding model is downloaded on first startup** (not baked into the image). Modal's `hf_embedding_cache` volume persists `/hf_cache` across cold starts, so the 90 MB download happens once per deployment, not once per cold start. Same pattern as the existing `main_modal.py`.
- **Secrets are injected via env vars** (Modal's `Secret.from_dict` on the wrapper, same env-var contract as the existing inline secret). Nothing secret ever enters the Dockerfile.
- **`.dockerignore`** excludes `.env`, `__pycache__`, `qdrant_storage/`, `mlruns/`, `test/`, `benchmarking/`, `docs/`, `.planning/`, and IDE/tooling state.

## Considered Options

**Image source**
- A: Modal builds from Dockerfile via `from_dockerfile` — simpler, no registry, but CI has no portable artefact.
- B: CI builds + pushes to registry, Modal pulls via `from_registry` — image is portable to any cloud, but needs a registry account and a `docker login` step.
- C (chosen): both supported via `IMAGE_TAG` env var. Local dev: A. CI: B. One wrapper script, one set of routes, two paths.

**GPU strategy**
- A: One CUDA image (~5 GB), CI must use a GPU runner.
- B: CPU-only image everywhere. Embedding model runs on CPU in prod, missing the `torch.cuda.is_available()` fast path.
- C (chosen): two separate files (`Dockerfile` CPU, `Dockerfile.cuda` GPU). CI uses the CPU one, Modal uses the CUDA one. Different files, intentionally.

**HF model weight in the image**
- A: Bake the 90 MB MiniLM into the image layer. First cold start is instant; model bump requires `docker build` + `docker push`.
- B (chosen): download on first startup, persist to the existing `hf_embedding_cache` Modal volume. `EMBED_MODEL_ID` is pinned in code so the model identity is deterministic; one-time 90 MB download per fresh deployment is acceptable; model bump is a one-line code change.

**App entry-point shape**
- A: Keep `scripts/main_modal.py` as the FastAPI source and have the wrapper import from it. Tight coupling; the FastAPI file still imports `modal`.
- B (chosen): New `scripts/main_docker.py` is a `modal`-free FastAPI module. New `scripts/main_modal_docker.py` is the Modal wrapper that imports `app` from it. The two existing scripts (`main_fastapi.py` for local uvicorn dev, `main_modal.py` for the legacy Modal build) stay untouched as reference / fallback paths.

## Consequences

- **Two ways to deploy, same FastAPI app.** `docker run` and `modal deploy` use the same `scripts.main_docker:app` object. The CI workflow is: `docker build -f Dockerfile.cuda -t $REGISTRY/cuda:$SHA . && docker push && IMAGE_TAG=$REGISTRY/cuda:$SHA modal deploy -m scripts.main_modal_docker`. A new dev's first deploy is just `modal deploy -m scripts.main_modal_docker` with no `IMAGE_TAG` set.
- **Routes live in two files.** `main_modal.py` (legacy Modal build) and `main_docker.py` (Docker build) both define the same routes. Adding a route means editing both until the legacy path is retired. The factory pattern (`scripts/_app_factory.py` shared by both) is the obvious follow-up refactor.
- **`main_docker.py` has a `modal` import for the volume commit.** The import is guarded by `try/except`, but the file is no longer 100% Modal-free at the module level. Anyone running `python -m scripts.main_docker` locally (without `modal` installed) sees the import skip and the volume commit become a no-op.
- **First cold start of a fresh deploy pays a 90 MB download.** Mitigated by the `hf_embedding_cache` volume: after the first cold start, the model is in the volume and subsequent cold starts are fast. Optional pre-warm: run a one-shot `modal` function that calls `embed_query("warmup")` and `volume.commit()` after the first deploy.
- **Local repro is now portable.** Anyone with Docker can run `docker build -f Dockerfile -t rag-book:dev . && docker run --env-file .env -p 8000:8000 rag-book:dev` and get a service that behaves like the Modal one (minus GPU/CUDA).
- **The cross-drive Windows bug is sidestepped**, not fixed. Anyone using the legacy `main_modal.py` path on the same Windows host still hits it; the fix lives in the deploy path, not in the SDK.
