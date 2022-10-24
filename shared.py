import concurrent.futures
import datetime
import json
import logging
import os
import subprocess
import sys
import traceback
from asyncio import get_event_loop, sleep
from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, HTTPServer
from inspect import getframeinfo, stack
from socketserver import ThreadingMixIn
from typing import (Any, Awaitable, Callable, Generic, Optional, Tuple, Type,
                    TypeVar, Union, cast)
from urllib import request
from urllib.error import URLError

from genericpath import exists

T = TypeVar("T")
R = TypeVar("R")
Fn = Callable[[T], R]

paracetamol_smiles = "CC(=O)Nc1ccc(O)cc1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tensorflow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
project_dir = os.path.dirname(os.path.realpath(__file__))
logger = logging.Logger(__name__)
conda_dir = os.environ["CONDA_PREFIX"]


class Timer:
    @staticmethod
    def calc(fn: Callable[[], T]) -> Tuple[float, T]:
        timer = Timer()
        result = fn()
        return timer.done(), result

    @staticmethod
    async def acalc(awaitable: Awaitable[T]) -> Tuple[float, T]:
        timer = Timer()
        result = await awaitable
        return timer.done(), result

    def __init__(self):
        self.start = datetime.datetime.now()

    def done(self):
        self.delta = (datetime.datetime.now() - self.start).total_seconds()
        return self.delta


def _serve(port: int, callback: Fn[Any, Any]):
    class S(BaseHTTPRequestHandler):
        def log_message(self, *args: Any):
            pass

        def do_POST(self):
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length).decode("utf-8")
            try:
                in_data = json.loads(post_data)
                out_data = callback(in_data)
                result = (False, out_data)
            except Exception as e:
                result = (
                    True,
                    "".join(
                        traceback.format_exception(type(e), value=e, tb=e.__traceback__)
                    ),
                )
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        """Handle requests in a separate thread."""

    httpd = ThreadedHTTPServer(("localhost", port), S)
    httpd.serve_forever()


def serve(callback: Fn[Any, Any]):
    _serve(int(sys.argv[1]), callback)


def disable_mf():
    if os.environ.get("DISABLE_MF"):
        caller = getframeinfo(stack()[1][0])
        print(f"DISABLE_MF: {caller.filename}:{caller.lineno}")
        return True
    else:
        return False


def disable_syba():
    if os.environ.get("DISABLE_SYBA"):
        caller = getframeinfo(stack()[1][0])
        print(f"DISABLE_SYBA: {caller.filename}:{caller.lineno}")
        return True
    else:
        return False


class AppKilled(Exception):
    pass


class AppError(Exception):
    pass


def _is_errno(e: URLError, errno: int):
    return isinstance(e.reason, OSError) and e.reason.errno == errno


def _fetch(port: int, data: Any, remaining_retries: int = 5) -> Tuple[bool, Any]:
    def f(e: Exception):
        if remaining_retries > 0:
            return _fetch(port, data, remaining_retries - 1)
        else:
            return (True, e)

    try:
        req = request.Request(
            f"http://localhost:{port}",
            method="POST",
            data=json.dumps(data).encode(),
        )
        res = request.urlopen(req)
        return (False, json.loads(res.read()))
    except URLError as e:
        if _is_errno(e, 110):
            return f(e)
        else:
            raise e
    except ConnectionResetError as e:
        return f(e)
    except Exception as e:
        return (True, e)


class CondaApp(Generic[T, R]):
    def __init__(self, port: int, module: str, env: str):
        self.port = port
        self.module = module
        self.env = env
        self.p: Optional[subprocess.Popen[Any]] = None
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

    def __str__(self):
        return f'CondaApp(port={self.port}, module="{self.module}", env="{self.env}")'

    def running(self):
        return self.p and (self.p.poll() is None)

    def _process_fetch(self, x: Tuple[bool, Any]):
        is_internal_error, v = x
        if is_internal_error:
            logger.exception(v)
            raise v
        else:
            is_app_error, value = v
            if is_app_error:
                raise AppError(value)
            else:
                return value

    def fetch_sync(self, data: Optional[T]) -> R:
        if self.running():
            return self._process_fetch(_fetch(self.port, data))
        else:
            raise AppKilled()

    async def fetch(self, data: Optional[T]) -> R:
        if self.running():
            return self._process_fetch(
                await get_event_loop().run_in_executor(
                    self.executor, _fetch, self.port, data
                )
            )
        else:
            raise AppKilled()

    async def _start(self):
        assert not self.executor
        assert not self.p
        self.executor = concurrent.futures.ProcessPoolExecutor()
        self.p = subprocess.Popen(
            [
                os.path.join(conda_dir, "envs", self.env, "bin/python"),
                "-m",
                f"{self.module}.main",
                str(self.port),
            ],
            env={
                **os.environ,
                "PYTHONPATH": f"{os.environ.get('PYTHONPATH', '')}:{project_dir}/askcos-core/",
            },
        )
        while self.running():
            await sleep(1)
            try:
                await self.fetch(None)
                break
            except RemoteDisconnected:
                pass
            except URLError as e:
                if not _is_errno(e, 111):
                    raise e

    async def start(self):
        time, _ = await Timer.acalc(self._start())
        print(f"{self} starting time: {time}")

    async def __aenter__(self):
        await self.start()
        return self.fetch, self.fetch_sync

    def stop(self):
        if self.executor:
            self.executor.shutdown()
        self.executor = None
        if self.running():
            assert self.p
            self.p.kill()
            try:
                self.p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.p.send_signal(9)
            self.p = None

    async def __aexit__(self, *_: Any):
        self.stop()


class Db:
    def __init__(self, name: str, readonly: bool):
        from sqlitedict import SqliteDict
        dirname = f"{project_dir}/results"
        if not exists(dirname):
            os.mkdir(dirname)
        self.db = SqliteDict(
            f"{dirname}/{name}.sqlite",
            outer_stack=False,
            autocommit=True,
            flag="r" if readonly else "c",
        )

    def _write(self, raw_key: str, value: T) -> T:
        self.db[raw_key] = json.dumps(value, sort_keys=True)
        return value

    async def maybe_create(self, key: Any, f: Callable[[], Awaitable[Any]]):
        raw_key = json.dumps(key, sort_keys=True)
        if raw_key not in self.db:
            self._write(raw_key, await f())

    def read_or_create_sync(self, key: Any, f: Callable[[], T]) -> T:
        raw_key = json.dumps(key, sort_keys=True)
        return (
            cast(T, self._read(raw_key, type(T)))
            if raw_key in self.db
            else self._write(raw_key, f())
        )

    async def read_or_create(
        self, key: Any, f: Callable[[], Awaitable[T]]
    ) -> Awaitable[T]:

        raw_key = json.dumps(key, sort_keys=True)
        return (
            self.db[raw_key] if raw_key in self.db else self._write(raw_key, await f())
        )

    def _read(self, raw_key: Any, type: Type[T]) -> Union[T, None]:
        return cast(type, json.loads(self.db.get(raw_key, "null")))

    def read(self, key: Any, type: Type[T]) -> Union[T, None]:
        return self._read(json.dumps(key, sort_keys=True), type)

    def write(self, key: Any, value: Any):
        self._write(json.dumps(key, sort_keys=True), value)

    def as_json(self):
        return json.dumps({k: v for k, v in self.db.iteritems()}, indent=2)

    def __enter__(self):
        self.db.__enter__()
        return self

    def __exit__(self, *exc_info):
        self.db.close()
