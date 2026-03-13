"""Embedder Engine — Dedicated Modal CPU container for Sentence Transformers.

Hosts the Qwen3-Embedding-0.6B model on CPU via SentenceTransformers.
This isolates the zero-trust embedding compute to a remote Modal service.
"""

from typing import List

import modal

from fleet_app import app  # shared app defined in app.py

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# ---------------------------------------------------------------------------
# Container image — CPU + SentenceTransformers
# ---------------------------------------------------------------------------

embedder_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sentence-transformers>=2.0.0",
        "huggingface-hub",
    )
)


@app.cls(
    image=embedder_image,
    cpu=1.0,
    keep_warm=0,
    timeout=600,
    retries=0,
)
class Embedder:
    """A remote embedding service using SentenceTransformers."""

    @modal.enter()
    def load_model(self):
        from sentence_transformers import SentenceTransformer
        # Load the model directly into memory
        self.model = SentenceTransformer(MODEL_NAME)

    @modal.method()
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts into embeddings."""
        # encode returns numpy arrays or tensors, so we convert to python lists
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
