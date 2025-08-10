
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from uuid import uuid4
import threading, time

# ====== Engine ======

@dataclass
class ActionRecord:
    id: str
    concept: str
    action: str
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]]
    flow: str
    t: float = field(default_factory=lambda: time.time())

class Concept:
    def __init__(self, name: str):
        self.name = name
    def perform(self, action: str, input_map: Dict[str, Any]) -> Dict[str, Any]:
        fn = getattr(self, action, None)
        if fn is None:
            raise AttributeError(f"{self.name}.{action} not found")
        return fn(**input_map)
    def query(self, qname: str, input_map: Dict[str, Any]) -> Dict[str, Any]:
        if not qname.startswith("_"):
            raise ValueError("Query names must start with '_' to be pure")
        fn = getattr(self, qname, None)
        if fn is None:
            raise AttributeError(f"{self.name}.{qname} not found")
        return fn(**input_map)

Var = str
@dataclass
class WhenPattern:
    concept: str
    action: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Var] = field(default_factory=dict)
WhereFn = Callable[["Engine", "Frame"], bool]
ThenFn = Callable[["Frame"], List[Tuple[str, str, Dict[str, Any]]]]
@dataclass
class Sync:
    name: str
    when: List[WhenPattern]
    where: Optional[WhereFn]
    then: ThenFn
class Frame:
    def __init__(self, flow: str, actions: List[ActionRecord]):
        self.flow = flow
        self.actions = actions
        self.vars: Dict[str, Any] = {}
    def bind(self, var: Var, value: Any) -> None:
        self.vars[var] = value
    def get(self, var: Var) -> Any:
        return self.vars[var]
    def try_get(self, var: Var, default: Any=None) -> Any:
        return self.vars.get(var, default)
    def last(self, concept: str, action: str) -> Optional[ActionRecord]:
        for rec in reversed(self.actions):
            if rec.concept == concept and rec.action == action:
                return rec
        return None
class Engine:
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.syncs: List[Sync] = []
        self.flow_log: Dict[str, List[ActionRecord]] = {}
        self._lock = threading.Lock()
        # Deduplicate actions emitted within the same flow to avoid infinite sync loops
        self._emitted: Dict[str, set] = {}
    def register_concept(self, concept: Concept) -> None:
        self.concepts[concept.name] = concept
    def register_sync(self, sync: Sync) -> None:
        self.syncs.append(sync)
    def start_flow(self) -> str:
        return str(uuid4())
    def invoke(self, concept: str, action: str, input_map: Dict[str, Any], *, flow: Optional[str]=None) -> ActionRecord:
        with self._lock:
            if flow is None:
                flow = self.start_flow()
            rec_id = str(uuid4())
            output = self.concepts[concept].perform(action, input_map)
            rec = ActionRecord(id=rec_id, concept=concept, action=action, input=input_map, output=output, flow=flow)
            self.flow_log.setdefault(flow, []).append(rec)
        # Evaluate outside lock (we call back into engine) but protect reads inside
        self._evaluate_syncs(flow)
        return rec
    def query(self, concept: str, qname: str, **kwargs) -> Dict[str, Any]:
        return self.concepts[concept].query(qname, kwargs)
    def _evaluate_syncs(self, flow: str) -> None:
        import json
        made_progress = True
        while made_progress:
            made_progress = False
            with self._lock:
                actions = list(self.flow_log[flow])
                emitted = self._emitted.setdefault(flow, set())
            frame = Frame(flow, actions)
            for sync in self.syncs:
                if self._match_when(sync, frame):
                    if sync.where is None or sync.where(self, frame):
                        for concept, action, params in sync.then(frame):
                            # Deduplicate identical (concept, action, params) within this flow
                            key = (concept, action, json.dumps(params, sort_keys=True, default=str))
                            with self._lock:
                                if key in emitted:
                                    continue
                                emitted.add(key)
                            self.invoke(concept, action, params, flow=flow)
                            made_progress = True

    def _match_when(self, sync: Sync, frame: Frame) -> bool:
        for pat in sync.when:
            rec = frame.last(pat.concept, pat.action)
            if rec is None or rec.output is None:
                return False
            for k, expected in pat.inputs.items():
                actual = rec.input.get(k)
                if isinstance(expected, str) and expected.startswith("$"):
                    frame.bind(expected[1:], actual)
                else:
                    if actual != expected:
                        return False
            for ok, var in pat.outputs.items():
                if ok not in rec.output:
                    return False
                frame.bind(var, rec.output[ok])
        return True