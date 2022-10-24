import json
from asyncio import run
from sys import argv
from typing import Callable, Coroutine

from shared import Db

from .ai import ai_input_gen
from .helpers import app_ai, app_scorers
from .tree import Tree
from .types import AiTree, Timed

methods: dict[str, Callable[[], Coroutine[None, None, None]]] = {}


def use(fn: Callable[[], Coroutine[None, None, None]]):
    methods[fn.__name__] = fn
    return fn


@use
async def ai():
    async with app_ai() as (fetch, _):
        for db, ai_input, _mol in ai_input_gen(False, False):
            await db.maybe_create(["ai", ai_input], lambda: fetch(ai_input))


@use
async def ai_postprocess():
    async with app_scorers() as (_, smileser):
        for db, ai_input, _mol in ai_input_gen(False, False):
            timed_ai_tree = db.read(["ai", ai_input], Timed[AiTree])
            if timed_ai_tree:
                _, ai_tree = timed_ai_tree

                async def f():
                    tree = await Tree.from_ai(ai_tree, smileser)
                    return tree.json()

                await db.maybe_create(["ai_postprocess", ai_input], lambda: f())
            else:
                print("error")


@use
async def cat_db():
    with Db(argv[2], True) as db:
        print(db.as_json())


@use
async def ra_scores():
    with Db("scores", True) as db:
        for k, v in db.db.iteritems():
            key = json.loads(k)
            if isinstance(key, list) and key[0] == "ra":
                val = json.loads(v)
                if val < 0.0 or val > 1.0:
                    print(val)


if __name__ == "__main__":
    method = methods.get(argv[1])
    if method:
        run(method())
    else:
        print(methods.keys())
