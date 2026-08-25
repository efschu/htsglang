from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import requests
import zmq

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import sock_recv, sock_send, wrap_as_pickle
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import get_free_port, get_zmq_socket_on_host
from sglang.test.scripted_runtime.io_struct import (
    HookReady,
    OutOfBandError,
    RunScript,
    ScriptFailed,
    ScriptSucceeded,
    Shutdown,
)
from sglang.test.scripted_runtime.utils import close_zmq_socket

DEFAULT_RUN_TIMEOUT_S: float = 120.0
SHUTDOWN_JOIN_TIMEOUT_S: float = 60.0
LISTENER_ACCEPT_TIMEOUT_S: float = 300.0
HTTP_READY_TIMEOUT_S: float = 300.0
HTTP_READY_POLL_INTERVAL_S: float = 0.5
SERVER_HOST: str = "127.0.0.1"


class ScriptedHttpServer:

    def __init__(
        self,
        *,
        ctx: zmq.Context,
        socket: zmq.Socket,
        server_process: mp.process.BaseProcess,
        out_of_band_error_path: Path,
        http_port: int,
    ) -> None:
        self._ctx = ctx
        self._socket = socket
        self._server_process = server_process
        self._out_of_band_error_path = out_of_band_error_path
        self._base_url = f"http://{SERVER_HOST}:{http_port}"
        self._shutdown_done = False
        self._dirty: Optional[str] = None

    @classmethod
    def start(cls, **engine_kwargs: Any) -> ScriptedHttpServer:
        out_of_band_error_path = _create_oob_error_file()

        ctx = zmq.Context()
        dispatch_port, socket = get_zmq_socket_on_host(ctx, zmq.PAIR, host=SERVER_HOST)
        server_process, http_port = _spawn_server_process(
            endpoint=f"tcp://{SERVER_HOST}:{dispatch_port}",
            out_of_band_error_path=out_of_band_error_path,
            engine_kwargs=engine_kwargs,
        )

        self = cls(
            ctx=ctx,
            socket=socket,
            server_process=server_process,
            out_of_band_error_path=out_of_band_error_path,
            http_port=http_port,
        )
        try:
            self._await_handshake()
            self._await_http_ready()
        except BaseException:
            self._teardown()
            raise
        return self

    def execute_script(
        self,
        script_fn: Callable,
        *,
        args: Tuple[Any, ...] = (),
        timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
    ) -> None:
        if self._dirty:
            raise RuntimeError(f"ScriptedHttpServer is dirty: {self._dirty}")

        fn_path = f"{script_fn.__module__}:{script_fn.__qualname__}"
        sock_send(self._socket, wrap_as_pickle(RunScript(fn_path=fn_path, args=args)))

        if not self._socket.poll(int(timeout_s * 1000)):
            if not self._server_process.is_alive():
                self._dirty = f"server process died before responding to {fn_path!r}"
                raise RuntimeError(self._dirty)
            self._dirty = f"script {fn_path!r} timed out after {timeout_s}s"
            raise TimeoutError(self._dirty)

        reply = sock_recv(self._socket)
        match reply:
            case ScriptFailed(traceback=tb):
                raise AssertionError(f"scripted-runtime script failed:\n{tb}")
            case ScriptSucceeded():
                return
            case _:
                raise RuntimeError(
                    f"scheduler hook replied with unexpected message {reply!r}"
                )

    def shutdown(self) -> None:
        if self._shutdown_done:
            return

        fatal_error: Optional[OutOfBandError] = None
        try:
            try:
                sock_send(self._socket, wrap_as_pickle(Shutdown()))
            except zmq.ZMQError:
                pass

            self._server_process.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
            self._terminate_process(self._server_process)
            fatal_error = self._read_out_of_band_error()
        finally:
            self._teardown()
            self._shutdown_done = True

        if fatal_error:
            raise AssertionError(
                f"scripted-runtime server failed:\n{fatal_error.traceback}"
            )

    def _await_handshake(self) -> None:
        # #861h: POLL IN SLICES AND CHECK THE CHILD EVERY SLICE.
        #
        # THE HANG, py-spy'd on 2026-08-25: test_dirty_session_refuses_to_run
        # sat in this call at 0 % CPU while its spawned server child
        # (pid 2502190) was already <defunct> -- dead at startup, never reaped.
        # A single blocking `poll(LISTENER_ACCEPT_TIMEOUT_S)` cannot notice
        # that: it waits out the whole deadline for a handshake that can never
        # arrive, and on a long deadline that reads as a wedged test run rather
        # than as a failed server. "Aktiv aber tot" (#369 family), in the test
        # infrastructure itself.
        #
        # A deadline alone is not the fix -- it was already here. The missing
        # half is LIVENESS: a peer that has exited will never speak, so the
        # wait must end the moment the child does, and it must say WHY.
        deadline = time.monotonic() + LISTENER_ACCEPT_TIMEOUT_S
        slice_ms = 250
        while True:
            if self._socket.poll(slice_ms):
                break
            # multiprocessing.Process, NOT subprocess.Popen -- `.poll()` would
            # AttributeError here. `execute_script` at :95 already uses the
            # right accessor; matching it is what keeps ONE liveness idiom in
            # this file instead of two.
            if not self._server_process.is_alive():
                rc = self._server_process.exitcode
                raise RuntimeError(
                    f"ScriptedHttpServer: the spawned server child (pid "
                    f"{self._server_process.pid}) EXITED with code {rc} before "
                    f"the handshake. Waiting for a handshake from a dead peer "
                    f"is the hang this check exists to prevent.\n"
                    f"--- child stderr ---\n{self._drain_child_stderr()}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ScriptedHttpServer: HTTP server did not connect within "
                    f"{LISTENER_ACCEPT_TIMEOUT_S}s (child pid "
                    f"{self._server_process.pid} still alive)"
                )

        ready = sock_recv(self._socket)
        if not isinstance(ready, HookReady):
            raise RuntimeError(
                f"ScriptedHttpServer: expected HookReady handshake, got {ready!r}"
            )

    def _drain_child_stderr(self) -> str:
        """Whatever the child managed to say before dying. #861h.

        CAPTURED, because "the child died" without WHY moves the hang one
        question later. Best-effort and never raises: this runs on a failure
        path and must not replace the failure it is explaining.
        """
        for attr in ("stderr", "stdout"):
            stream = getattr(self._server_process, attr, None)
            if stream is None:
                continue
            try:
                data = stream.read()
            except Exception:  # noqa: BLE001 - a diagnostic never raises
                continue
            if not data:
                continue
            if isinstance(data, bytes):
                data = data.decode("utf-8", "replace")
            return data[-4000:]
        try:
            err = self._read_out_of_band_error()
        except Exception:  # noqa: BLE001
            err = None
        if err is not None:
            return str(getattr(err, "traceback", err))[-4000:]
        return "(no child output captured)"

    def _await_http_ready(self) -> None:
        # HookReady only means the scheduler dispatch loop started; the uvicorn
        # entrypoint may not be bound yet. The first script runs
        # _reset_engine_state, which POSTs to this server's own HTTP port, so
        # block until the port is bound and routing before any script can run.
        #
        # Poll /model_info rather than /health: once the server reports Up,
        # /health runs the generation-based health check (a real probe request
        # through the scheduler), which cannot complete while the scheduler
        # waits for scripts between RunScript commands. Any /model_info
        # response proves the socket is bound and routes are registered, which
        # is all the control POSTs need.
        url = f"{self._base_url}/model_info"
        deadline = time.monotonic() + HTTP_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if not self._server_process.is_alive():
                raise RuntimeError(
                    "ScriptedHttpServer: server process died during HTTP startup"
                )
            try:
                requests.get(url, timeout=2.0)
                return
            except requests.RequestException:
                pass
            time.sleep(HTTP_READY_POLL_INTERVAL_S)

        raise TimeoutError(
            f"ScriptedHttpServer: HTTP endpoint {url} not bound within "
            f"{HTTP_READY_TIMEOUT_S}s"
        )

    def _teardown(self) -> None:
        try:
            close_zmq_socket(self._socket, self._ctx)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        self._terminate_process(self._server_process)
        self._cleanup_files()

    def _read_out_of_band_error(self) -> Optional[OutOfBandError]:
        try:
            text = self._out_of_band_error_path.read_text()
        except OSError:
            return None
        text = text.strip()
        return OutOfBandError.from_json(text) if text else None

    def _cleanup_files(self) -> None:
        try:
            self._out_of_band_error_path.unlink()
        except OSError:
            pass

    @staticmethod
    def _terminate_process(process: mp.process.BaseProcess) -> None:
        if not process.is_alive():
            return
        process.terminate()
        process.join(timeout=10.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=10.0)


def _create_oob_error_file() -> Path:
    err_fd, err_path = tempfile.mkstemp(
        prefix="scripted_runtime_oob_error_", suffix=".json"
    )
    os.close(err_fd)
    return Path(err_path)


def _spawn_server_process(
    *,
    endpoint: str,
    out_of_band_error_path: Path,
    engine_kwargs: Dict[str, Any],
) -> Tuple[mp.process.BaseProcess, int]:
    mp_ctx = mp.get_context("spawn")
    launch_kwargs: Dict[str, Any] = dict(
        host=SERVER_HOST,
        port=get_free_port(),
        kv_canary="raise",
        kv_canary_real_data="partial",
        kv_canary_sweep_interval=100,
        disable_prefill_cuda_graph=True,
    )
    launch_kwargs.update(engine_kwargs)
    http_port = launch_kwargs["port"]
    server_process = mp_ctx.Process(
        target=_launch_scripted_http_server,
        kwargs=launch_kwargs,
        name="scripted-runtime-http-server",
        daemon=False,
    )

    sys_path_entry = str(Path(__file__).resolve().parent)
    with (
        envs.SGLANG_TEST_SCRIPTED_RUNTIME.override(True),
        envs.SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR.override(endpoint),
        envs.SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH.override(
            str(out_of_band_error_path)
        ),
        envs.SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY.override(sys_path_entry),
    ):
        server_process.start()

    return server_process, http_port


def _launch_scripted_http_server(**engine_kwargs: Any) -> None:
    launch_server(ServerArgs(**engine_kwargs))
