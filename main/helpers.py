import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

from shared import CondaApp

from .score import Score, Smiles
from .types import AiInput, AiTree, Scoring, Timed

all_scorings: list[Scoring] = ["sa", "sc", "ra", "syba", "mf"]


@asynccontextmanager
async def app_scorers():
    async with CondaApp[Tuple[str, List[str]], List[float]](
        8002, "scorers", "scorers"
    ) as (fetch, _):

        async def f(scoring: Scoring, smiles: str):
            return (await fetch((scoring, [smiles])))[0]

        async def g(data: Tuple[str, Optional[int]]):
            s, t = data
            sa, sc, ra, syba, mf = await asyncio.gather(
                f("sa", s), f("sc", s), f("ra", s), f("syba", s), f("mf", s)
            )
            return Smiles(s, Score(sa=sa, sc=sc, ra=ra, mf=mf, syba=syba), t)

        yield f, g


@asynccontextmanager
async def app_ai():
    async with CondaApp[AiInput, Timed[AiTree]](8001, "ai", "aizynth-env") as x:
        yield x
