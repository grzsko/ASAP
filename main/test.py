import asyncio
import json
from typing import Any, Awaitable, Callable, Optional, Tuple
from unittest import IsolatedAsyncioTestCase, main

from shared import paracetamol_smiles

from .ai import zero_setup
from .helpers import app_ai, app_scorers
from .score import Score, Smiles
from .tree import Tree
from .types import Scoring


class Test(IsolatedAsyncioTestCase):
    def setUp(self):
        self.failed: list[str] = []

    def tearDown(self):
        if self.failed:
            self.fail("".join(self.failed))

    async def diff(
        self,
        scoring: Scoring,
        expected: float,
        smiles: str,
        scorer: Callable[[Scoring, str], Awaitable[float]],
    ):
        actual = await scorer(scoring, smiles)
        if 0.001 < abs(expected - actual):
            self.failed.append(f"\n{scoring} {expected=} {actual=} {smiles=}")

    async def test_ai(self):
        async with app_ai() as (ai, _):
            _, ai_tree = await ai({"setup": zero_setup, "smiles": paracetamol_smiles})

            async def fake_scorer(x: Tuple[str, Optional[int]]):
                smiles, transforms = x
                return Smiles(smiles, Score(0.0, 0.0, 0.0, 0.0, 0.0), transforms)

            await Tree.from_ai(ai_tree, fake_scorer)

    async def test_sa_sc_ra_syba_mf(self):
        with open(f"main/test.json") as f:
            mols: list[dict[str, Any]] = json.load(f)
            async with app_scorers() as (scorer, _):
                subjects: list[tuple[Scoring, float, str]] = [
                    *(
                        (scoring, m[scoring], m["smiles"])
                        for scoring in ("sa", "sc", "ra", "syba")
                        for m in mols
                    ),
                    ("mf", 0.508526768404158, "O=C(C)Oc1ccccc1C(=O)O"),
                    (
                        "mf",
                        0.241901002275,
                        "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
                    ),
                ]
                await asyncio.gather(
                    *[self.diff(*subject, scorer) for subject in subjects]
                )


if __name__ == "__main__":
    main()
