"""DRS-based execution scope tracking (Honest DRT Implementation).

Maps Kamp's Discourse Representation Theory onto sandboxed task execution.
This acts as the agent's 'mental model' of the workspace.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, Any
from pydantic import BaseModel, Field, ConfigDict


@dataclass(frozen=True)
class Referent:
    """A DRT Discourse Referent: an entity introduced into the universe."""
    ref_id: str
    referent_type: str  # e.g., 'file', 'variable'
    path_or_name: str
    produced_by: str    # Task ID


class Witness(BaseModel):
    """An MLTT proof term: proves a proposition is satisfied via state evidence."""
    model_config = ConfigDict(frozen=True)
    
    witness_id: str
    witness_type: str
    value_hash: str        # Cryptographic proof (SHA256) of sandbox state
    proposition: str       # The type this proves (e.g. "file_exists")
    context_gamma: str     # The Task ID serving as the proof environment


@dataclass(frozen=True)
class Condition:
    """A DRT Condition: a formal predicate defining relations between referents."""
    predicate: str
    args: tuple


@dataclass
class DRS:
    """A Discourse Representation Structure — an internal model of the execution scope."""
    refs: set[Union[Referent, Witness]] = field(default_factory=set)
    conditions: set[Condition] = field(default_factory=set)
    parent_label: Optional[str] = field(default=None)
    label: str = "main"

    def certify_witness(self, witness: Witness) -> bool:
        """MLTT: Formally unify a proof term with the discourse context (Gamma)."""
        if any(isinstance(r, Witness) and r.witness_id == witness.witness_id for r in self.refs):
            return False
        self.refs.add(witness)
        self.conditions.add(Condition("proved_at", (witness.witness_id, witness.value_hash)))
        return True

    def introduce(self, task_id: str, referent_type: str, path_or_name: str) -> Referent:
        """DRT: Introduce a new referent into the discourse universe."""
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

    def introduce_from_delta(self, task_id: str, delta: Any) -> None:
        """Frege-DRT Bridge: Update beliefs based on observed side effects (StateDelta)."""
        for path in delta.created:
            self.introduce(task_id, "file", path)
        for path in delta.modified:
            # Modify conditions track the history of an existing referent
            self.conditions.add(Condition("modified_by", (path, task_id)))

    def resolve(self, description: str, parent_drs: Optional["DRS"] = None) -> list[Referent]:
        """Anaphora Resolution: Link linguistic descriptions to accessible referents."""
        acc = self.refs.copy()
        if parent_drs is not None:
            acc |= parent_drs.refs
        
        matched = []
        for r in acc:
            if not isinstance(r, Referent):
                continue
            name = r.path_or_name
            basename = name.split("/")[-1] if "/" in name else name
            if basename in description or name in description:
                matched.append(r)
        return matched

    def augment_description(self, description: str, parent_drs: Optional["DRS"] = None) -> str:
        """Resolves vague references ('the file') to concrete accessible referents."""
        resolved = self.resolve(description, parent_drs)
        if not resolved:
            return description
        context = ", ".join(f"{r.referent_type}:{r.path_or_name}" for r in resolved[:5])
        return f"{description}\n[Discourse Context (Accessible): {context}]"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "parent_label": self.parent_label,
            "refs": [
                r.model_dump() if isinstance(r, Witness) else {
                    "ref_id": r.ref_id,
                    "referent_type": r.referent_type,
                    "path_or_name": r.path_or_name,
                    "produced_by": r.produced_by,
                    "_type": "Referent"
                } for r in self.refs
            ],
            "conditions": [{"predicate": c.predicate, "args": list(c.args)} for c in self.conditions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DRS":
        drs = cls(label=data.get("label", "main"), parent_label=data.get("parent_label"))
        for r in data.get("refs", []):
            if r.get("_type") == "Referent":
                r.pop("_type")
                drs.refs.add(Referent(**r))
            else:
                drs.refs.add(Witness(**r))
        for c in data.get("conditions", []):
            drs.conditions.add(Condition(c["predicate"], tuple(c["args"])))
        return drs
