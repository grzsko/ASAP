from main.data import data
from shared import Db

from .types import AiInput, Setup

zero_setup: Setup = {
    "score": "sc",
    "uw_multiplier": 0.0,
    "agg": "max",
}

# 5 * 3 * 3 + 1 = 46 setups per molecule
ai_setups: list[Setup] = [
    zero_setup,
    *[
        {
            "score": score,
            "uw_multiplier": uw_multiplier,
            "agg": agg,
        }
        for score in ("sa", "sc", "ra", "syba", "mf")
        for uw_multiplier in (0.2375, 0.475, 0.7125)  # 0.95/4 * 1, *2, *3,
        for agg in ("min", "max", "avg")
    ],
]


def ai_input_gen(readonly: bool, only_zero_setup: bool):
    with Db("db", readonly) as db:
        mols = data()
        count = len(mols)
        for mol_no, mol in enumerate(mols):
            if only_zero_setup:
                ai_input: AiInput = {"smiles": mol.smiles, "setup": zero_setup}
                yield db, ai_input, mol
            else:
                for setup_no, setup in enumerate(ai_setups):
                    print(
                        f"[ai][setup {setup_no}/{len(ai_setups)}][mol {mol_no}/{count}]"
                    )
                    ai_input: AiInput = {"smiles": mol.smiles, "setup": setup}
                    yield db, ai_input, mol
