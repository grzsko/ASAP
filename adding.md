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
The score should be rescaled so that it fits to the range [0,1], where 0 is a non-synthesizable molecule and 1 means an easily-synthesizable molecule.
For our tiny example dividing by 2 was enough, but other operations can be applied, such as inverting, translating, truncating, etc.
4. A new score with its name should be registered in the `scorers` dict located in the same file:
```python
scorers: Dict[str, SmilesScorer] = {
        "ra": get_ra_scorer("DNN", "chembl"),
        "sa": wrap_to_mol(get_sa_scorer()),
        "sc": wrap_to_mol(get_sc_scorer()),
        "syba": wrap_to_mol(get_syba_scorer()),
        "tiny": get_foo_scorer(),
    }
```
Function `wrap_to_mol` is applied for several scores which require as an input a RdKit `Mol` object.
It applies an additional layer of transforming input Smiles into `Mol` object.
5. For correct working, the score name has to be added in several places of the code.
