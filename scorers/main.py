import os
import sys
from typing import Dict, List, NewType, Optional, Tuple, TypeVar

from rdkit import Chem

from shared import Db, Fn, paracetamol_smiles, project_dir, serve

Mol = NewType("Mol", object)
T = TypeVar("T")
SmilesScorer = Fn[str, float]
MolScorer = Fn[Mol, float]

to_mol: Fn[str, Mol] = Chem.MolFromSmiles
to_smiles: Fn[Mol, str] = Chem.MolToSmiles


def inverted(f: Fn[float, float]) -> Fn[float, float]:
    return lambda value: 1.0 - f(value)


def wrap_to_mol(f: Fn[Mol, T]) -> Fn[str, T]:
    return lambda smiles: f(to_mol(smiles))


def truncate(value: float, lowest: float, highest: float):
    # [-inf, inf] -> [lowest, highest]
    return min(max(value, lowest), highest)


def scaler(lowest: float, highest: float) -> Fn[float, float]:
    """
    Returns scale function that transforms linearly value, where value in [lowest, highest], to [0,1]
    0 means infeasible (not-accessible) molecule
    1 means fully feasible molecule
    """
    # [-inf, inf] -> [lowest, highest] -> [0, highest - lowest] -> [0, 1]
    l, h = float(min(lowest, highest)), float(max(lowest, highest))
    size = h - l
    return lambda value: (truncate(value, l, h) - l) / size


def get_mf_scorer() -> SmilesScorer:
    from .mf_score.mfscore.ocsvm import score_smiles

    def f(smiles: str):
        # [-inf, inf] -> [-800, 600] -> [-1, 1] -> [-0.5, 0.5] -> [0, 1]
        value = truncate(score_smiles(smiles), -800.0, 600.0)
        return (value / 2.0 / (800.0 if value < 0.0 else 600.0)) + 0.5

    return f


def get_sc_scorer() -> MolScorer:
    from .scscore_numpy import SCScorer

    sc = SCScorer()
    sc.restore()
    scale = inverted(scaler(1, 5))
    return lambda mol: scale(sc.apply(sc.mol_to_fp(sc, mol)))


def get_syba_scorer() -> MolScorer:
    print("Loading syba scorer. It's gonna take ~2 minutes.")
    from syba.syba import SybaClassifier

    syba = SybaClassifier()
    syba.fitDefaultScore()
    scale = scaler(-100, 100)
    return lambda mol: scale(syba.predict(mol=mol))


def get_ra_scorer(
    model: str,  # Literal["DNN", "XGB"],
    db: str,  # Literal["chembl", "gdbchembl", "gdbmedchem"]
) -> SmilesScorer:
    from RAscore import RAscore_NN, RAscore_XGB

    dnn = model == "DNN"
    x = "fcfp" if dnn else "ecfp"
    y = "h5" if dnn else "pkl"
    model = (RAscore_NN.RAScorerNN if dnn else RAscore_XGB.RAScorerXGB)(
        f"{project_dir}/data/ra_models/{model}_{db}_{x}_counts/model.{y}"
    )
    return lambda smiles: truncate(model.predict(smiles).item(), 0.0, 1.0)


def get_sa_scorer() -> MolScorer:
    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

    f = inverted(scaler(1, 10))
    return lambda mol: f(sascorer.calculateScore(mol))


if __name__ == "__main__":
    scorers: Dict[str, SmilesScorer] = {
        "ra": get_ra_scorer("DNN", "chembl"),
        "sa": wrap_to_mol(get_sa_scorer()),
        "sc": wrap_to_mol(get_sc_scorer()),
        "mf": get_mf_scorer(),
        "syba": wrap_to_mol(get_syba_scorer()),
    }
    smileser = wrap_to_mol(to_smiles)
    with Db("scores", False) as db:

        def get(type: str, smiles: str):
            smiles_canon = db.read_or_create_sync(
                ["smiles", smiles], lambda: smileser(smiles)
            )
            return db.read_or_create_sync(
                [type, smiles_canon], lambda: scorers[type](smiles_canon)
            )

        def scorer(data: Optional[Tuple[str, List[str]]]):
            if data:
                type, smileses = data
                return [get(type, smiles) for smiles in smileses]

        for key in scorers.keys():
            scorer((key, [paracetamol_smiles]))  # preprocess
            scorer((key, [paracetamol_smiles]))  # read

        serve(scorer)
