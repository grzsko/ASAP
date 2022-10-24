from typing import Literal, Optional, TypedDict, TypeVar

T = TypeVar("T")
Timed = tuple[float, T]
Scoring = Literal["sa", "sc", "ra", "syba", "mf"]


class AiTree(TypedDict):
    is_solved: bool
    ai_score: float
    expandable: list[tuple[str, Optional[int]]]
    in_stock: list[tuple[str, Optional[int]]]
    children: list[Optional["AiTree"]]


class Setup(TypedDict):
    agg: Literal["max", "min", "avg"]
    score: Scoring
    uw_multiplier: float


class AiInput(TypedDict):
    smiles: str
    setup: Setup
