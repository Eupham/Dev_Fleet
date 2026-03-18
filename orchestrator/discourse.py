"""DRS-based execution scope tracking (Honest DRT)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, Any
from pydantic import BaseModel, ConfigDict

@dataclass(frozen=True)
class Referent:
    ref_id: str
    referent_type: str
    path_or_name: str
    produced_by: str

class Witness(BaseModel):
    """MLTT Proof Term."""
    model_config = ConfigDict(frozen=True)
    witness_id: str
    witness_type: str
    value_hash: str
    proposition: str
    context_gamma: str

@dataclass(frozen=True)
class Condition:
    predicate: str
    args: tuple

@dataclass
class DRS:
    """Discourse Representation Structure — Agent's Mental Model."""
    refs: set[Union[Referent, Witness]] = field(default_factory=set)
    conditions: set[Condition] = field(default_factory=set)
    parent_label: Optional[str] = field(default=None)
    label: str = "main"

    def introduce(self, task_id: str, referent_type: str, path_or_name: str) -> Referent:
        ref = Referent(ref_id=f"{task_id}:{path_or_name}", referent_type=referent_type, 
                       path_or_name=path_or_name, produced_by=task_id)
        self.refs.add(ref)
        self.conditions.add(Condition("created_by", (ref.ref_id, task_id)))
        return ref

    def introduce_from_delta(self, task_id: str, delta: Any) -> None:
        for path in delta.created: self.introduce(task_id, "file", path)
        for path in delta.modified: self.conditions.add(Condition("modified_by", (path, task_id)))

    def augment_description(self, description: str, parent_drs: Optional["DRS"] = None) -> str:
        acc = self.refs.copy()
        if parent_drs: acc |= parent_drs.refs
        matched = [r for r in acc if isinstance(r, Referent) and (r.path_or_name in description)]
        if not matched: return description
        context = ", ".join(f"{r.referent_type}:{r.path_or_name}" for r in matched[:5])
        return f"{description}\n[Discourse Context: {context}]"

    def open_retry_scope(self, task_id: str) -> "DRS":
        return DRS(refs=set(), conditions=set(), parent_label=self.label, label=f"retry:{task_id}")

    def commit_retry_scope(self, retry_drs_dict: dict) -> None:
        if retry_drs_dict.get("parent_label") == self.label:
            retry = DRS.from_dict(retry_drs_dict)
            self.refs.update(retry.refs)
            self.conditions.update(retry.conditions)

    def to_dict(self) -> dict:
        return {
            "label": self.label, "parent_label": self.parent_label,
            "refs": [r.model_dump() if isinstance(r, Witness) else {**r.__dict__, "_type": "Referent"} for r in self.refs],
            "conditions": [{"predicate": c.predicate, "args": list(c.args)} for c in self.conditions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DRS":
        drs = cls(label=data.get("label", "main"), parent_label=data.get("parent_label"))
        for r in data.get("refs", []):
            if r.get("_type") == "Referent":
                r.pop("_type"); drs.refs.add(Referent(**r))
            else: drs.refs.add(Witness(**r))
        for c in data.get("conditions", []):
            drs.conditions.add(Condition(c["predicate"], tuple(c["args"])))
        return drs
