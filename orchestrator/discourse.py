# orchestrator/discourse.py
"""DRS-based execution scope tracking.

Maps Kamp's Discourse Representation Theory onto sandboxed task execution:

  DRT concept          → Execution concept
  ─────────────────────────────────────────
  Discourse referent   → File or variable produced by a task
  DRS universe (refs)  → Set of files in scope at this point
  DRS conditions       → Predicates: created_by(ref, task), path(ref, name)
  Subordinate DRS box  → Retry scope (child of the outer execution scope)
  Accessibility        → A retry can read outer scope's files; its own
                         introductions are isolated until commit or discard
  Anaphora resolution  → augment_description() resolves vague references
                         (e.g. "the output file") to concrete accessible paths

Serialization contract:
  to_dict() / from_dict() survive the AgentState JSON round-trip.
  The live parent DRS object is not kept post-deserialization; callers
  must pass it explicitly to accessible_refs_with_parent() and
  augment_description(). commit_retry_scope() verifies by parent_label
  string match rather than object identity for the same reason.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
from pydantic import BaseModel, Field
class Witness(BaseModel):
    """An MLTT proof term: proves a postcondition is satisfied via a hash."""
    witness_id: str
    witness_type: str
    value_hash: str        # Cryptographic proof of sandbox state
    proposition: str       # The type this proves (e.g. "file_exists")
    context_gamma: str     # The Task ID serving as the proof environment


@dataclass(frozen=True)
class Condition:
    """A predicate on referents."""
    predicate: str
    args: tuple


@dataclass
class DRS:
    """A Discourse Representation Structure — one execution scope."""
    refs: set = field(default_factory=set)
    conditions: set = field(default_factory=set)
    parent_label: Optional[str] = field(default=None)
    label: str = "main"

    def certify_witness(self, witness: Witness) -> bool:
        """Formally unify a proof term with the discourse context."""
        if any(r.witness_id == witness.witness_id for r in self.refs if isinstance(r, Witness)):
            return False
        self.refs.add(witness)
        self.conditions.add(Condition("proved_at", (witness.witness_id, witness.value_hash)))
        return True

    def introduce(self, task_id: str, referent_type: str, path_or_name: str):
        """Add a referent to this scope."""
        ref = Referent(
            ref_id=f"{task_id}:{path_or_name}",
            referent_type=referent_type,
            path_or_name=path_or_name,
            produced_by=task_id,
        )
        self.refs.add(ref)
        self.conditions.add(Condition("created_by", (ref.ref_id, task_id)))
        self.conditions.add(Condition("path", (ref.ref_id, path_or_name)))
        return ref

    def introduce_from_delta(self, task_id: str, delta) -> None:
        """Add all files from a completed task's StateDelta to this scope."""
        for path in delta.created:
            self.introduce(task_id, "file", path)
        for path in delta.modified:
            self.conditions.add(Condition("modified_by", (path, task_id)))

    def accessible_refs(self) -> set:
        """Return all referents accessible from this scope.

        Includes own refs. Parent refs must be passed in separately
        since the live parent object is not kept after deserialization.
        Use accessible_refs_with_parent() when the parent DRS is available.
        """
        return set(self.refs)

    def accessible_refs_with_parent(self, parent_drs: "DRS") -> set:
        """Return own refs plus parent's refs.

        Call this when augmenting task descriptions with accessible files.
        The parent DRS is the outer scope's DRS, looked up by label from state.
        """
        return self.refs | parent_drs.refs

    def resolve(self, description: str, parent_drs: Optional["DRS"] = None) -> list:
        """Find accessible referents whose path appears in description."""
        acc = self.refs.copy()
        if parent_drs is not None:
            acc |= parent_drs.refs
        matched = []
        for ref in acc:
            name = ref.path_or_name
            basename = name.split("/")[-1] if "/" in name else name
            if basename in description or name in description:
                matched.append(ref)
        return matched

    def augment_description(
        self, description: str, parent_drs: Optional["DRS"] = None
    ) -> str:
        """Prepend accessible file referents to a task description.

        Called before reranking. Resolves anaphoric references like
        'the output file' to concrete paths accessible in this scope.
        """
        resolved = self.resolve(description, parent_drs)
        if not resolved:
            return description
        context = ", ".join(
            f"{r.referent_type}:{r.path_or_name}" for r in resolved[:5]
        )
        return f"{description}\n[Accessible files: {context}]"

    def open_retry_scope(self, task_id: str) -> "DRS":
        """Open a child scope for a retry attempt.

        The child stores this DRS's label as parent_label so that
        commit_retry_scope can verify the relationship after
        deserialization (without needing the live parent object).
        """
        return DRS(
            refs=set(),
            conditions=set(),
            parent_label=self.label,
            label=f"retry:{task_id}",
        )

    def commit_retry_scope(self, retry_drs_dict: dict) -> None:
        """Merge a successful retry's referents into this scope.

        Verifies by parent_label string match — not object identity —
        so this survives the AgentState JSON round-trip.

        Args:
            retry_drs_dict: the serialized retry DRS from AgentState.
        """
        if retry_drs_dict.get("parent_label") != self.label:
            raise ValueError(
                f"commit_retry_scope: retry DRS parent_label "
                f"{retry_drs_dict.get('parent_label')!r} does not match "
                f"outer DRS label {self.label!r}."
            )
        retry_drs = DRS.from_dict(retry_drs_dict)
        self.refs.update(retry_drs.refs)
        self.conditions.update(retry_drs.conditions)

    def discard_retry_scope(self) -> None:
        """Discard a failed retry scope. No-op by design.

        The retry's refs were never in this DRS. Nothing to remove.
        The retry DRS dict is cleared from AgentState by the caller.
        """
        pass

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "parent_label": self.parent_label,
            "refs": [
                {
                    "ref_id": r.ref_id,
                    "referent_type": r.referent_type,
                    "path_or_name": r.path_or_name,
                    "produced_by": r.produced_by,
                }
                for r in self.refs
            ],
            "conditions": [
                {"predicate": c.predicate, "args": list(c.args)}
                for c in self.conditions
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DRS":
        drs = cls(
            label=data.get("label", "main"),
            parent_label=data.get("parent_label"),
        )
        for r in data.get("refs", []):
            drs.refs.add(Referent(**r))
        for c in data.get("conditions", []):
            drs.conditions.add(Condition(c["predicate"], tuple(c["args"])))
        return drs
