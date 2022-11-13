import inspect
from typing import Any, Callable, Iterable, Optional, TypedDict, TypeVar, Union

T = TypeVar("T")


def not_none(l: Iterable[Optional[T]]) -> Iterable[T]:
    return (e for e in l if e is not None)


def flatten(l: Iterable[Iterable[T]]) -> Iterable[T]:
    return (e for sl in l for e in sl)


def serialize_dict(d: Union[dict[str, Any], TypedDict], sep: str):
    return sep.join([f"{k}: {v}" for k, v in d.items()])


def human_json(d: Union[dict[str, Any], TypedDict]):
    return "{" + serialize_dict(d, ", ") + "}"


def fn_txt(fn: Callable[[Any], Any]):
    return fn.__name__
    # return inspect.getsource(fn).replace("\n", "")
