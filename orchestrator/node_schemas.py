# orchestrator/node_schemas.py
"""Typed schemas for knowledge graph nodes.

Required fields are those the agent needs to operate:
  - SemanticNode: concept_type (drives retrieval classification)
  - ProceduralNode: actor_capability + implementation_depth (drive routing)
  - EpisodicNode: description + task_type (drive execution)

Optional fields add context for the LLM and the graph viewer.
ValidationError on a required field means the node cannot be used for
its intended purpose and should not be added.
"""
from __future__ import annotations
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field


class _NodeBase(BaseModel):
    node_id: str
    content: str                         # required — one-sentence summary
    label: str                           # required — assigned by labeler
    graph_type: Literal["semantic", "procedural", "episodic"]


# ── Semantic graph schemas ────────────────────────────────────────────────

class ConceptNode(_NodeBase):
    """A code concept: data structure, algorithm, pattern, interface, etc."""
    frame_name: Literal["Code_Concept"] = "Code_Concept"
    graph_type: Literal["semantic"] = "semantic"
    concept_type: Literal[
        "data_structure", "algorithm", "pattern",
        "interface", "protocol", "invariant"
    ]                                    # required
    source_file: str = ""
    language: str = "python"
    complexity_class: str = ""


class ModuleNode(_NodeBase):
    """A file, package, or component."""
    frame_name: Literal["Code_Module"] = "Code_Module"
    graph_type: Literal["semantic"] = "semantic"
    module_path: str                     # required
    exports: list[str] = []
    depends_on: list[str] = []


SemanticNode = Annotated[
    Union[ConceptNode, ModuleNode],
    Field(discriminator="frame_name"),
]


# ── Procedural graph schemas ──────────────────────────────────────────────

class ExecutionNode(_NodeBase):
    """A known execution pattern: how to accomplish a class of task."""
    frame_name: Literal["Task_Execution"] = "Task_Execution"
    graph_type: Literal["procedural"] = "procedural"
    actor_capability: Literal["bash", "python", "llm_only"]      # required
    implementation_depth: Literal["algorithm", "library", "syscall"]  # required
    execution_cost: Literal["trivial", "moderate", "expensive"] = "moderate"
    idempotent: bool = False
    side_effects: list[str] = []


class ConstraintNode(_NodeBase):
    """A constraint, requirement, or limit that applies to execution."""
    frame_name: Literal["Execution_Constraint"] = "Execution_Constraint"
    graph_type: Literal["procedural"] = "procedural"
    constraint_type: Literal[
        "memory_limit", "latency_limit", "api_contract",
        "type_contract", "security_requirement"
    ]                                    # required
    enforced_by: str                     # required
    threshold: str = ""
    recovery_strategy: str = ""


ProceduralNode = Annotated[
    Union[ExecutionNode, ConstraintNode],
    Field(discriminator="frame_name"),
]


# ── Episodic graph schema ─────────────────────────────────────────────────

class EpisodicNode(_NodeBase):
    """A single executed task episode."""
    frame_name: Literal["Task_Episode"] = "Task_Episode"
    graph_type: Literal["episodic"] = "episodic"
    description: str                     # required
    task_type: Literal[
        "query", "transform", "verify", "compose"
    ] = "transform"                      # required
    status: str = "pending"
    difficulty: float = 0.0
    preconditions: list[str] = []
    postconditions: list[str] = []
    actor_capability: Literal["bash", "python", "llm_only"] = "python"
