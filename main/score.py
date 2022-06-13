from typing import Optional, Tuple, TypedDict

from main.data import Db
from shared import Fn

from .utils import serialize_dict


class JsonScore(TypedDict):
    sa: float
    sc: float
    ra: float
    mf: float
    syba: float


class Score:
    def __init__(self, sa: float, sc: float, ra: float, mf: float, syba: float):
        self.sa = sa
        self.sc = sc
        self.ra = ra
        self.mf = mf
        self.syba = syba

    def __str__(self):
        return serialize_dict(self.json(), ",")

    @staticmethod
    def from_json(j: JsonScore):
        return Score(sa=j["sa"], sc=j["sc"], ra=j["ra"], syba=j["syba"], mf=j["mf"])

    @staticmethod
    def getters() -> list[Tuple[str, Fn["Score", float]]]:
        return [
            ("sa", lambda s: s.sa),
            ("sc", lambda s: s.sc),
            ("ra", lambda s: s.ra),
            ("mf", lambda s: s.mf),
            ("syba", lambda s: s.syba),
        ]

    def json(self) -> JsonScore:
        return {
            "sa": self.sa,
            "sc": self.sc,
            "ra": self.ra,
            "mf": self.mf,
            "syba": self.syba,
        }


class JsonSmiles(TypedDict):
    smiles: str
    score: JsonScore
    transforms: Optional[int]


class Smiles:
    cache: dict[str, Score] = {}

    def __init__(self, smiles: str, score: Score, transforms: Optional[int]):
        self.smiles = smiles
        self.score = score
        self.transforms = transforms

    def __str__(self):
        return serialize_dict(self.json(), ", ")

    def json(self) -> JsonSmiles:
        return {
            "smiles": self.smiles,
            "score": self.score.json(),
            "transforms": self.transforms,
        }

    @classmethod
    def from_db(cls, smiles: str, transforms: Optional[int], db: Db):
        def f(scoring: str):  # TODO Scoring type
            ret = db.read([scoring, smiles])
            assert isinstance(ret, float)
            return ret

        def g():
            s = Score(f("sa"), f("sc"), f("ra"), f("mf"), f("syba"))
            Smiles.cache[smiles] = s
            return s

        return Smiles(smiles, Smiles.cache.get(smiles) or g(), transforms)

    @staticmethod
    def from_json(j: JsonSmiles):
        return Smiles(j["smiles"], Score.from_json(j["score"]), j["transforms"])
