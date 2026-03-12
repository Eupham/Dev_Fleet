"""Reranker — Qwen3-Reranker-0.6B cross-encoder for edge scoring.

Assesses relevance between task nodes (from the DAG) and knowledge-graph
nodes so that Frege's compositionality principle can derive
complexity/difficulty via contextual edges.

The model outputs a binary yes/no judgment, converted to a [0, 1]
relevance score via log-softmax over "yes" and "no" token logits.
"""

from __future__ import annotations

import modal

from fleet_app import app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINUTES = 60  # seconds
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
MAX_LENGTH = 8192

RERANK_INSTRUCTION = (
    "Given a coding task, assess whether the knowledge-graph node "
    "contains information relevant to completing the task"
)

# ---------------------------------------------------------------------------
# Container image — PyTorch + Transformers (CPU-only, 0.6B model)
# ---------------------------------------------------------------------------

reranker_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_python_source("fleet_app", copy=True)
    .pip_install(
        "torch>=2.0",
        "transformers>=4.51.0",
        "huggingface-hub>=0.20",
    )
)

# Cache model weights across container restarts
reranker_cache_vol = modal.Volume.from_name(
    "reranker-cache-vol", create_if_missing=True
)

# ---------------------------------------------------------------------------
# Reranker class
# ---------------------------------------------------------------------------


@app.cls(
    image=reranker_image,
    volumes={"/root/.cache/huggingface": reranker_cache_vol},
    scaledown_window=5 * MINUTES,
    timeout=5 * MINUTES,
)
@modal.concurrent(max_inputs=50)
class Reranker:
    """Qwen3-Reranker-0.6B cross-encoder for graph-edge scoring."""

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL, padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL, torch_dtype=torch.float32,
        ).eval()
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        prefix = (
            '<|im_start|>system\nJudge whether the Document meets the '
            'requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".'
            '<|im_end|>\n<|im_start|>user\n'
        )
        suffix = (
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        self.prefix_tokens = self.tokenizer.encode(
            prefix, add_special_tokens=False,
        )
        self.suffix_tokens = self.tokenizer.encode(
            suffix, add_special_tokens=False,
        )

    @modal.method()
    def score_pairs(
        self,
        task_description: str,
        candidate_descriptions: list[str],
        instruction: str | None = None,
    ) -> list[float]:
        """Score *task_description* against each candidate.

        Returns a list of relevance scores in [0, 1].
        """
        import torch

        instruction = instruction or RERANK_INSTRUCTION
        pairs = [
            f"<Instruct>: {instruction}\n"
            f"<Query>: {task_description}\n"
            f"<Document>: {doc}"
            for doc in candidate_descriptions
        ]

        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=MAX_LENGTH - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens

        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=MAX_LENGTH,
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
            true_vec = logits[:, self.token_true_id]
            false_vec = logits[:, self.token_false_id]
            stacked = torch.stack([false_vec, true_vec], dim=1)
            scores = torch.nn.functional.log_softmax(stacked, dim=1)
            scores = scores[:, 1].exp().tolist()

        return scores
