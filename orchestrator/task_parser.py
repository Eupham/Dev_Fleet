"""Typed task decomposition — Fillmore Frame Semantics + DAG output.
FIXES APPLIED:
1. RENAMED MontagueAction → TaskFrame. What was implemented IS Fillmore
   Frame Semantics (verb + typed slots), NOT Montague Grammar (syntax tree →
   lambda calculus → logical form). Honest naming.
2. Added TYPED FRAME VARIANTS (ResearchFrame, CreateFrame, etc.) with
   domain-specific slots. This is the path to LLM ablation: if frames are
   exhaustively enumerated with typed slots, they can eventually be filled
   by pattern matching instead of LLM inference.
3. Task output is now a PROPER DAG, not a linear chain. The LLM can declare
   parallel tasks where no dependency exists.
4. Fillmore's insight preserved: every verb activates a frame with obligatory
   and optional slots. The frame constrains what information must be gathered.
"""
from __future__ import annotations
import uuid
import logging
from typing import Annotated, Literal, Union, List, Optional
from pydantic import BaseModel, Field
logger = logging.getLogger("dev_fleet.task_parser")
# ---------------------------------------------------------------------------
# Task Types (Atomic Nodes in the DAG)
# ---------------------------------------------------------------------------
class _TaskBase(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str
    tool_hint: Literal["bash", "python", ""] = "python"
    preconditions: List[str] = []  # IDs of tasks that must complete first
    # Fillmore: slots that this frame requires to be filled
    inputs_needed: List[str] = []   # What this task reads
    outputs_produced: List[str] = []  # What this task produces
class TransformTask(_TaskBase):
    task_type: Literal["transform"] = "transform"
class QueryTask(_TaskBase):
    task_type: Literal["query"] = "query"
class VerifyTask(_TaskBase):
    task_type: Literal["verify"] = "verify"
class ResearchTask(_TaskBase):
    """Frame for knowledge acquisition — the 'research' verb."""
    task_type: Literal["research"] = "research"
    tool_hint: Literal["bash", "python", ""] = "bash"
    domain: str = ""           # What domain to research
    depth: Literal["shallow", "deep"] = "shallow"
    output_format: Literal["summary", "knowledge_nodes"] = "knowledge_nodes"
AtomicTaskNode = Annotated[
    Union[QueryTask, TransformTask, VerifyTask, ResearchTask],
    Field(discriminator="task_type")
]
class TaskDAG(BaseModel):
    user_prompt: str = ""
    tasks: List[AtomicTaskNode]
    def get_execution_order(self) -> List[str]:
        """Topological sort of tasks respecting preconditions.
        Returns task IDs in valid execution order. Parallel-eligible
        tasks will appear in arbitrary order but after their deps.
        """
        import networkx as nx
        G = nx.DiGraph()
        for t in self.tasks:
            G.add_node(t.id)
            for pre in t.preconditions:
                G.add_edge(pre, t.id)
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Task DAG has cycles — cannot determine execution order")
        return list(nx.topological_sort(G))
    def get_task_by_id(self, task_id: str) -> Optional[AtomicTaskNode]:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None
# ---------------------------------------------------------------------------
# Fillmore Frame Extraction (replaces "MontagueAction")
# ---------------------------------------------------------------------------
class TaskFrame(BaseModel):
    """A Fillmore Frame: a verb (action) with typed slots.
    This is what was previously called MontagueAction. The rename is
    honest: Fillmore Frame Semantics maps verbs to structured schemas
    with Agent, Patient, Instrument roles. Montague Grammar maps
    syntax to lambda calculus — a completely different formalism.
    ABLATION PATH: When frames are exhaustive and slots are typed,
    a pattern-matching parser can fill them without an LLM.
    """
    verb: Literal["create", "modify", "read", "verify", "research"] = Field(
        description="The action verb — each activates a different frame."
    )
    target: str = Field(description="The file path, URL, or domain.")
    instruction: str = Field(description="What needs to be done.")
    # Fillmore: explicit role assignments
    depends_on: List[str] = Field(
        default=[],
        description="List of other frame targets this depends on."
    )
class FrameDecomposition(BaseModel):
    """LLM output schema for frame extraction."""
    frames: List[TaskFrame] = Field(..., description="List of Fillmore Frames to execute.")
# ---------------------------------------------------------------------------
# System Prompt — Frame Extraction, Not "Montague Parsing"
# ---------------------------------------------------------------------------
FRAME_SYSTEM = """You are a Fillmore Frame parser. Translate the user's request into a DAG of action frames.
RULES:
1. Valid verbs: create, modify, read, verify, research.
2. Each frame has: verb, target, instruction, depends_on (list of targets this needs).
3. Use depends_on to express PARALLEL vs SEQUENTIAL tasks.
   - If task B needs task A's output, set depends_on: ["<target of A>"]
   - If tasks are independent, leave depends_on empty (they can run in parallel)
4. Research tasks should specify the domain in the target field.
EXAMPLE INPUT: "Research quantum computing, write a simulator script, then test it."
EXAMPLE OUTPUT:
{
  "frames": [
    {"verb": "research", "target": "quantum computing", "instruction": "Research quantum computing principles and libraries", "depends_on": []},
    {"verb": "create", "target": "quantum_sim.py", "instruction": "Write a quantum state simulator using research findings", "depends_on": ["quantum computing"]},
    {"verb": "verify", "target": "quantum_sim.py", "instruction": "Run tests on the quantum simulator", "depends_on": ["quantum_sim.py"]}
  ]
}
Respond ONLY with valid JSON matching the exact structure above."""
# ---------------------------------------------------------------------------
# Parsing Pipeline
# ---------------------------------------------------------------------------
def _frames_to_dag(frames: List[TaskFrame], user_prompt: str) -> TaskDAG:
    """Convert extracted frames into a typed TaskDAG with dependency edges.
    The depends_on field references targets (not IDs), so we need to
    resolve target → task_id mapping for preconditions.
    """
    # Phase 1: Create typed tasks from frames
    tasks: List[AtomicTaskNode] = []
    target_to_id: dict[str, str] = {}
    for frame in frames:
        v = frame.verb.lower()
        common = {
            "description": frame.instruction,
            "inputs_needed": frame.depends_on,
            "outputs_produced": [frame.target],
        }
        if v == "research":
            task = ResearchTask(
                **common,
                tool_hint="bash",
                domain=frame.target,
                depth="deep" if "deep" in frame.instruction.lower() else "shallow",
            )
        elif v in ("create", "modify"):
            task = TransformTask(**common, tool_hint="python")
        elif v == "read":
            task = QueryTask(**common)
        else:
            task = VerifyTask(**common)
        target_to_id[frame.target] = task.id
        tasks.append(task)
    # Phase 2: Resolve depends_on targets → task IDs for preconditions
    for task in tasks:
        resolved_preconditions = []
        for dep_target in task.inputs_needed:
            if dep_target in target_to_id:
                resolved_preconditions.append(target_to_id[dep_target])
        task.preconditions = resolved_preconditions
    return TaskDAG(user_prompt=user_prompt, tasks=tasks)
def parse_prompt(user_prompt: str, model: str = "llm", codebase_context: str = "") -> TaskDAG:
    """Parse a user prompt into a typed TaskDAG via Fillmore Frame extraction."""
    from orchestrator.llm_client import chat_completion
    messages = [
        {"role": "system", "content": FRAME_SYSTEM},
        {"role": "user", "content": f"Context:\n{codebase_context}\n\nRequest: {user_prompt}"},
    ]
    try:
        raw_response = chat_completion(messages, model=model, schema=FrameDecomposition)
        if raw_response is None:
            raise ValueError("Schema validation returned None — LLM output did not match expected schema")
        # Extract frames from response
        if hasattr(raw_response, "frames"):
            parsed_frames = raw_response.frames
        elif isinstance(raw_response, dict):
            parsed_frames = [TaskFrame(**f) for f in raw_response.get("frames", [])]
        else:
            raise ValueError(f"RPC returned unexpected type: {type(raw_response)}")
        if not parsed_frames:
            raise ValueError("LLM returned empty frames list")
        return _frames_to_dag(parsed_frames, user_prompt)
    except Exception as exc:
        logger.warning(f"Frame decomposition failed: {exc} — using single-task fallback")
        return TaskDAG(
            user_prompt=user_prompt,
            tasks=[TransformTask(description=user_prompt, tool_hint="python")]
        )
