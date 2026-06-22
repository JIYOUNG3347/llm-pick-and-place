from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm_manip.contracts import WorldState
from llm_manip.env.base import RawObs


@runtime_checkable
class Perception(Protocol):
    def perceive(self, raw: RawObs) -> WorldState: ...
