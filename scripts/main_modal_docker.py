"""
Modal deployment entry point that uses the Docker image defined by
`Dockerfile.cuda` instead of Modal's local `pip install` walker.

Two deploy modes, selected by environment:

  IMAGE_TAG unset  ->  Modal builds `./Dockerfile.cuda` remotely (local dev)
  IMAGE_TAG=ghcr.io/me/rag-book:cuda-abc123  ->  Modal pulls the pre-built image
                                                  (CI / registry mode)

The FastAPI app itself lives in `scripts/main_docker.py` — this file is
pure Modal plumbing.

Usage:
    # Local dev — Modal builds the Dockerfile on its own builder
    modal deploy -m scripts.main_modal_docker

    # CI — push the image first, then deploy with the tag
    docker build -f Dockerfile.cuda -t ghcr.io/me/rag-book:cuda-$GITHUB_SHA .
    docker push ghcr.io/me/rag-book:cuda-$GITHUB_SHA
    IMAGE_TAG=ghcr.io/me/rag-book:cuda-$GITHUB_SHA modal deploy -m scripts.main_modal_docker
"""

import os

import modal

# ---------------- IMAGE ----------------
_image_tag = os.environ.get("IMAGE_TAG")
if _image_tag:
    # Registry mode: image is already built and pushed by CI.
    # The Dockerfile installs `modal` in requirements_main.txt, so the image's
    # own Python can import the modal client. add_python=None lets Modal
    # auto-detect that Python already exists.
    image = modal.Image.from_registry(_image_tag, add_python=None)
else:
    # Dockerfile mode: Modal builds the image remotely from this repo.
    # `context_dir="."` makes the build context the project root, so
    # COPY paths in the Dockerfile resolve correctly.
    image = modal.Image.from_dockerfile(
        "./Dockerfile.cuda",
        context_dir=".",
        add_python=None,
    )


# ---------------- SECRET ----------------
# Same env-var contract as the inline `Secret.from_dict` in scripts/main_modal.py.
# Values are read from the deployer's environment at import time.
_raw_secret = {
    "LLM_PROVIDER": os.getenv("LLM_PROVIDER"),
    "LLM_MODEL": os.getenv("LLM_MODEL"),
    "HUGGINGFACEHUB_API_TOKEN": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    "MINIMAX_API_KEY": os.getenv("MINIMAX_API_KEY"),
    "SUPABASE_DB_URL": os.getenv("SUPABASE_DB_URL"),
    "API_KEY": os.getenv("API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
    "QDRANT_ENDPOINT": os.getenv("QDRANT_ENDPOINT"),
    "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
    "SUPABASE_URL": os.getenv("SUPABASE_URL"),
    "SUPABASE_SERVICE_KEY": os.getenv("SUPABASE_SERVICE_KEY"),
    "SUPABASE_STORAGE_BUCKET": os.getenv("SUPABASE_STORAGE_BUCKET"),
    "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
    "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING"),
    "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT"),
}
_secret_dict = {k: v for k, v in _raw_secret.items() if v is not None}
hf_secrets = modal.Secret.from_dict(_secret_dict)


# ---------------- VOLUME ----------------
HF_CACHE_PATH = "/hf_cache"
embedding_cache_volume = modal.Volume.from_name("hf_embedding_cache")


# ---------------- APP ----------------
app = modal.App("book_qa_app_docker", image=image, secrets=[hf_secrets])


@app.function(
    timeout=2 * 3600,
    gpu="T4",
    image=image,
    secrets=[hf_secrets],
    volumes={HF_CACHE_PATH: embedding_cache_volume},
)
@modal.asgi_app(label="book-qa-fastapi")
def fastapi_app():
    # The FastAPI app is defined in main_docker.py — both `docker run` and
    # `modal deploy` use the same object, which is what reproducibility is
    # about. The import path assumes the image's PYTHONPATH includes /app
    # (set in both Dockerfiles).
    from scripts.main_docker import app as fastapi_app
    return fastapi_app
