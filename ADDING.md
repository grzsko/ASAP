# Adding a new synthetic accessibility score

This project intends to be a framework for comparing novel synthesizability scores. Thus, it was designed so that adding additional scores is relatively simple.

1. This project requires several differently configured Conda environments. If you need to add any specific dependencies required by a synthetic accessibility score, put them in the [Scorers Conda environment file](conda_scorers.yml).
2. Suppose that we want to add a new score, named `tiny`, which predicts synthetic accessibility by randomizing a number 0, 1, or 2, where 0 means infeasible (non-synthesizable) molecule and 2 means easily-synthesizable molecule.
3. Add score entry point. Usually, it is done by adding a function in [scorers/main.py](scorers/main.py) file, e.g.
    ```python
    def get_tiny_scorer() -> SmilesScorer:
        import numpy as np
        return lambda smiles: np.random.randint(3) / 2
    ```
Here, function get_tiny_scorer returns a function of type `SmilesScorer` which for a single argument being a molecule SMILES returns its score.
The score should be rescaled so that it fits the range [0,1], where 0 is a non-synthesizable molecule and 1 means an easily-synthesizable molecule.
For our tiny example dividing by 2 was enough, but other operations can be applied, such as inverting, translating, truncating, etc.

4. A new score with its name should be registered in the `scorers` dict located in the same file:
```python
scorers: Dict[str, SmilesScorer] = {
        "ra": get_ra_scorer("DNN", "chembl"),
        "sa": wrap_to_mol(get_sa_scorer()),
        "sc": wrap_to_mol(get_sc_scorer()),
        "syba": wrap_to_mol(get_syba_scorer()),
        "tiny": get_tiny_scorer(),
    }
```
Function `wrap_to_mol` is applied for several scores which require as an input a RdKit `Mol` object.
It applies an additional layer of transforming input Smiles into `Mol` object.

5. For correct score calling, the information of using a `tiny` score has to be added in several places of the code as listed below:
- [ai/main.py](ai/main.py) line 22, class Setup:
```python
class Setup(TypedDict):
    agg: Agg
    score: Literal["sa", "sc", "ra", "syba", "tiny"]
    uw_multiplier: float
```
- [main/ai.py](main/ai.py) line 21, defining all setups (proportions of exchanged score value in reward function):
```python
ai_setups: list[Setup] = [
    zero_setup,
    *[
        {
            "score": score,
            "uw_multiplier": uw_multiplier,
            "agg": agg,
        }
        for score in ("sa", "sc", "ra", "syba", "tiny")
        for uw_multiplier in (0.2375, 0.475, 0.7125)  # 0.95/4 * 1, *2, *3,
        for agg in ("min", "max", "avg")
    ],
]
```
- [main/helpers.py](main/helpers.py) line 8: `all_scorings: list[Scoring] = ["sa", "sc", "ra", "syba", "tiny"]`
- [main/helpers.py](main/helpers.py) lines 22-29:
```python
async def app_scorers():
    async with CondaApp[Tuple[str, List[str]], List[float]](
        8002, "scorers", "scorers"
    ) as (fetch, _):

        async def f(scoring: Scoring, smiles: str):
            return (await fetch((scoring, [smiles])))[0]

        async def g(data: Tuple[str, Optional[int]]):
            s, t = data
            sa, sc, ra, syba, tiny = await asyncio.gather(
                f("sa", s), f("sc", s), f("ra", s), f("syba", s), f("tiny", s)
            )
            return Smiles(s, Score(sa=sa, sc=sc, ra=ra, syba=syba, tiny=tiny), t)

        yield f, g
```
- [main/score.py](main/score.py) all entries in class score
- [main/types.py](main/types.py) line 5: `Scoring = Literal["sa", "sc", "ra", "syba", "tiny"]`
