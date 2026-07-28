# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Minimal ctypes binding for the UCX UCP tag-matching API.

Why ctypes and not ucx-py / a Cython extension
----------------------------------------------
The decisive requirement is not ergonomics, it is *which* ``libucp`` gets
loaded. UCX refuses to interoperate across releases whose UCP wire address
format differs -- a 1.18.1 peer talking to a 1.16.0 peer fails at endpoint
creation with the famously unhelpful ``invalid bandwidth 0.00``. The only
robust cure is version parity, which in a mixed fleet means "point this
process at the matching build", e.g. ``/opt/ucx116/lib/libucp.so.0``.

* ``ucx-py``/``ucp`` links (or bundles, via the RAPIDS ``libucx-*`` wheels)
  its *own* UCX. That does not solve the parity problem, it hard-codes one
  side of it, and it is asyncio-shaped -- the wrong control flow for a
  synchronous collective on the bs=1 decode critical path.
* A Cython extension needs a compiler and UCX *headers* on every host. The
  hosts in this fleet have the runtime libraries but not always the ``-dev``
  package, and building per host is exactly the drift we are trying to avoid.
* A subprocess bridge adds an IPC hop to a path whose entire point is ~1.5 us
  latency. Non-starter.

ctypes has none of those problems: no build step, no headers, and the library
to load is a plain string we can take from the environment. See
``SGLANG_HTCCL_UCX_LIB``.

ABI stability
-------------
Every UCP parameter struct is *mask-driven*: the callee reads only the fields
whose bit is set in ``field_mask`` / ``op_attr_mask``. New UCX releases append
fields and define new bits; the prefix layout of the fields we set has been
stable across the whole 1.x line. We additionally over-allocate every struct
(``_RESERVED``) so that even a callee which copies its own, larger struct
wholesale reads initialised zeroes rather than off the end of our buffer.
The offsets asserted at import time were dumped from UCX 1.16.0 headers.
"""

import ctypes
import functools
import os
import time

# ---------------------------------------------------------------------------
# Constants (values dumped from the UCX 1.16.0 headers; stable across 1.x)
# ---------------------------------------------------------------------------

UCS_OK = 0
UCS_INPROGRESS = 1
UCS_ERR_LAST = -100

UCP_PARAM_FIELD_FEATURES = 1 << 0
UCP_FEATURE_TAG = 1 << 0

UCP_WORKER_PARAM_FIELD_THREAD_MODE = 1 << 0
UCS_THREAD_MODE_SINGLE = 0
UCS_THREAD_MODE_MULTI = 2

UCP_EP_PARAM_FIELD_REMOTE_ADDRESS = 1 << 0
UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE = 1 << 1
UCP_ERR_HANDLING_MODE_PEER = 1

UCP_EP_CLOSE_FLAG_FORCE = 1 << 0

UCP_OP_ATTR_FIELD_CALLBACK = 1 << 1
UCP_OP_ATTR_FIELD_DATATYPE = 1 << 3
UCP_OP_ATTR_FIELD_MEMORY_TYPE = 1 << 6
UCP_OP_ATTR_FIELD_RECV_INFO = 1 << 7
UCP_OP_ATTR_FLAG_NO_IMM_CMPL = 1 << 16

UCS_MEMORY_TYPE_HOST = 0

# ucp_dt_make_contig(1); the default datatype, spelled out so the request
# parameters are fully explicit rather than relying on a zero default.
UCP_DATATYPE_CONTIG_1 = 8

# UCP API version we compile our expectations against. Passed to
# ucp_init_version so UCX itself can complain about an incompatible runtime.
UCP_API_MAJOR = 1
UCP_API_MINOR = 16

# Over-allocation for every parameter struct. UCX 1.16's largest is
# ucp_worker_params_t at 200 bytes; 1 KiB leaves room for a decade of appends.
_RESERVED = 1024


def _pad(used: int) -> int:
    return _RESERVED - used


class UcpParams(ctypes.Structure):
    _fields_ = [
        ("field_mask", ctypes.c_uint64),
        ("features", ctypes.c_uint64),
        ("_reserved", ctypes.c_ubyte * _pad(16)),
    ]


class UcpWorkerParams(ctypes.Structure):
    _fields_ = [
        ("field_mask", ctypes.c_uint64),
        ("thread_mode", ctypes.c_int),
        ("_reserved", ctypes.c_ubyte * _pad(12)),
    ]


class UcpEpParams(ctypes.Structure):
    _fields_ = [
        ("field_mask", ctypes.c_uint64),
        ("address", ctypes.c_void_p),
        ("err_mode", ctypes.c_int),
        ("_reserved", ctypes.c_ubyte * _pad(20)),
    ]


class UcpRequestParam(ctypes.Structure):
    """``ucp_request_param_t`` -- 72 bytes in 1.16, asserted below."""

    _fields_ = [
        ("op_attr_mask", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("request", ctypes.c_void_p),
        ("cb", ctypes.c_void_p),
        ("datatype", ctypes.c_uint64),
        ("user_data", ctypes.c_void_p),
        ("reply_buffer", ctypes.c_void_p),
        ("memory_type", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),
        ("recv_info", ctypes.c_void_p),
        ("memh", ctypes.c_void_p),
        ("_reserved", ctypes.c_ubyte * _pad(72)),
    ]


def _assert_layout() -> None:
    """Guard the hand-transcribed offsets against a typo.

    These are the offsets dumped from the 1.16.0 headers. A mismatch here is a
    coding error in this file, not a UCX problem -- fail at import, loudly.
    """
    expect = {
        "op_attr_mask": 0,
        "flags": 4,
        "request": 8,
        "cb": 16,
        "datatype": 24,
        "user_data": 32,
        "reply_buffer": 40,
        "memory_type": 48,
        "recv_info": 56,
        "memh": 64,
    }
    for name, off in expect.items():
        actual = getattr(UcpRequestParam, name).offset
        if actual != off:
            raise RuntimeError(
                f"htccl_ucx_bindings: ucp_request_param_t.{name} laid out at "
                f"{actual}, expected {off}. The struct definition in this file "
                f"is wrong."
            )
    for cls, name, off in (
        (UcpEpParams, "address", 8),
        (UcpEpParams, "err_mode", 16),
        (UcpWorkerParams, "thread_mode", 8),
        (UcpParams, "features", 8),
    ):
        actual = getattr(cls, name).offset
        if actual != off:
            raise RuntimeError(
                f"htccl_ucx_bindings: {cls.__name__}.{name} laid out at "
                f"{actual}, expected {off}."
            )


_assert_layout()


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------


class UcxError(RuntimeError):
    """A UCX call returned a failure status."""


class UcxVersionMismatch(RuntimeError):
    """Peers in the group run different UCX releases.

    Raised by the transport's handshake, defined here so callers can catch it
    without importing the transport.
    """


# ``UCS_PTR_IS_ERR`` threshold, precomputed. Every post compares against this,
# so it is a module constant rather than a c_size_t round trip per call.
_ERR_PTR_FLOOR = ctypes.c_size_t(UCS_ERR_LAST).value


def _ptr_is_err(p: int) -> bool:
    """``UCS_PTR_IS_ERR`` -- error "pointers" are small negative status codes."""
    return p >= _ERR_PTR_FLOOR


def _ptr_status(p: int) -> int:
    """Reinterpret an error pointer as its ``ucs_status_t``."""
    return p - (1 << 64)


class UcpLibrary:
    """The loaded ``libucp``, with prototypes declared.

    ``SGLANG_HTCCL_UCX_LIB`` selects the library. When it names an explicit
    path we also pre-load ``libucs``/``libuct`` from the *same directory* with
    ``RTLD_GLOBAL``, so pointing at a side-by-side install (the version-parity
    workaround, e.g. ``/opt/ucx116/lib/libucp.so.0``) works without the caller
    also having to get ``LD_LIBRARY_PATH`` right. Getting that half-right is
    precisely how a mismatched ``libucs`` sneaks back in.
    """

    _instance = None

    def __init__(self, path: str | None = None):
        self.path = path or os.environ.get("SGLANG_HTCCL_UCX_LIB", "libucp.so.0")
        if os.path.sep in self.path:
            directory = os.path.dirname(os.path.abspath(self.path))
            for sibling in ("libucs.so.0", "libuct.so.0"):
                candidate = os.path.join(directory, sibling)
                if os.path.exists(candidate):
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            self._ucs_path = os.path.join(directory, "libucs.so.0")
        else:
            self._ucs_path = "libucs.so.0"
        try:
            self.lib = ctypes.CDLL(self.path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            raise UcxError(
                f"htccl ucx: cannot load '{self.path}' ({e}). Install UCX "
                f"(apt: libucx0) or set SGLANG_HTCCL_UCX_LIB to a libucp.so.0."
            ) from e
        try:
            self.ucs = ctypes.CDLL(self._ucs_path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            self.ucs = self.lib  # status_string lives in libucs; degrade quietly
        self._declare()

    @classmethod
    def instance(cls) -> "UcpLibrary":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _declare(self) -> None:
        L = self.lib
        c_vp, c_szt, c_u64, c_int = (
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_int,
        )

        L.ucp_get_version.restype = None
        L.ucp_get_version.argtypes = [
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ]
        L.ucp_get_version_string.restype = ctypes.c_char_p
        L.ucp_get_version_string.argtypes = []

        L.ucp_config_read.restype = c_int
        L.ucp_config_read.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                      ctypes.POINTER(c_vp)]
        L.ucp_config_release.restype = None
        L.ucp_config_release.argtypes = [c_vp]
        # Present across the whole 1.x line. Declared unconditionally; the
        # caller only reaches it when a net device was requested explicitly.
        L.ucp_config_modify.restype = c_int
        L.ucp_config_modify.argtypes = [c_vp, ctypes.c_char_p, ctypes.c_char_p]

        L.ucp_init_version.restype = c_int
        L.ucp_init_version.argtypes = [
            ctypes.c_uint, ctypes.c_uint,
            ctypes.POINTER(UcpParams), c_vp, ctypes.POINTER(c_vp),
        ]
        L.ucp_cleanup.restype = None
        L.ucp_cleanup.argtypes = [c_vp]

        L.ucp_worker_create.restype = c_int
        L.ucp_worker_create.argtypes = [
            c_vp, ctypes.POINTER(UcpWorkerParams), ctypes.POINTER(c_vp)
        ]
        L.ucp_worker_destroy.restype = None
        L.ucp_worker_destroy.argtypes = [c_vp]
        L.ucp_worker_get_address.restype = c_int
        L.ucp_worker_get_address.argtypes = [
            c_vp, ctypes.POINTER(c_vp), ctypes.POINTER(c_szt)
        ]
        L.ucp_worker_release_address.restype = None
        L.ucp_worker_release_address.argtypes = [c_vp, c_vp]
        L.ucp_worker_progress.restype = ctypes.c_uint
        L.ucp_worker_progress.argtypes = [c_vp]

        L.ucp_ep_create.restype = c_int
        L.ucp_ep_create.argtypes = [
            c_vp, ctypes.POINTER(UcpEpParams), ctypes.POINTER(c_vp)
        ]
        L.ucp_ep_close_nbx.restype = c_vp
        L.ucp_ep_close_nbx.argtypes = [c_vp, ctypes.POINTER(UcpRequestParam)]

        L.ucp_tag_send_nbx.restype = c_vp
        L.ucp_tag_send_nbx.argtypes = [
            c_vp, c_vp, c_szt, c_u64, ctypes.POINTER(UcpRequestParam)
        ]
        L.ucp_tag_recv_nbx.restype = c_vp
        L.ucp_tag_recv_nbx.argtypes = [
            c_vp, c_vp, c_szt, c_u64, c_u64, ctypes.POINTER(UcpRequestParam)
        ]
        L.ucp_request_check_status.restype = c_int
        L.ucp_request_check_status.argtypes = [c_vp]
        L.ucp_request_free.restype = None
        L.ucp_request_free.argtypes = [c_vp]

        try:
            self.ucs.ucs_status_string.restype = ctypes.c_char_p
            self.ucs.ucs_status_string.argtypes = [c_int]
        except AttributeError:
            pass

    # -- helpers -----------------------------------------------------------

    def status_string(self, status: int) -> str:
        try:
            return self.ucs.ucs_status_string(status).decode()
        except Exception:
            return f"status {status}"

    def check(self, status: int, what: str) -> None:
        if status != UCS_OK:
            raise UcxError(f"htccl ucx: {what} failed: {self.status_string(status)}")

    def version(self) -> tuple[int, int, int]:
        major, minor, release = (ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint())
        self.lib.ucp_get_version(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(release)
        )
        return (major.value, minor.value, release.value)

    def version_string(self) -> str:
        return self.lib.ucp_get_version_string().decode()


# ---------------------------------------------------------------------------
# Object wrappers
# ---------------------------------------------------------------------------


# Receives match the full tag: every transfer is identified exactly by
# (seq, phase, src, chunk), so two concurrently posted receives can never be
# satisfied by each other's message.
_TAG_MASK_EXACT = ctypes.c_uint64(0xFFFFFFFFFFFFFFFF)


def _make_request_param(recv: bool = False) -> UcpRequestParam:
    """Request parameters for a poll-completed contiguous host transfer.

    No callback is registered on purpose: the caller polls
    ``ucp_request_check_status`` while progressing the worker. For a
    latency-critical collective that is both the simplest and the fastest
    control flow -- a Python callback trampoline on the completion path would
    cost more than it saves.

    ``MEMORY_TYPE = HOST`` is an assertion about HTCCL's design, not a guess:
    every byte this transport puts on the wire has already been staged into a
    host buffer (there is no GPUDirect on this hardware), so telling UCX so
    skips its memory-type detection on each call.
    """
    p = UcpRequestParam()
    p.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE
    p.datatype = UCP_DATATYPE_CONTIG_1
    p.memory_type = UCS_MEMORY_TYPE_HOST
    return p


class UcpWorker:
    """A UCP context + worker, plus persistent endpoints to peers.

    One worker per process. Endpoints are created once at rendezvous and kept
    for the lifetime of the transport -- the brief's "persistente UCX-Endpoints,
    keine per-Call-Verbindungen". Endpoint setup involves a wireup handshake
    that costs orders of magnitude more than the ~1.5 us the link is capable
    of; doing it per collective would dominate every decode step.
    """

    def __init__(
        self,
        lib: UcpLibrary | None = None,
        timeout_s: float = 300.0,
        net_devices: str | None = None,
    ):
        self.lib = lib or UcpLibrary.instance()
        self.timeout_s = timeout_s
        self.net_devices = net_devices or None
        L = self.lib.lib

        config = ctypes.c_void_p()
        # NULL env prefix -> the standard UCX_* environment variables apply
        # (UCX_TLS, UCX_NET_DEVICES, UCX_IB_GID_INDEX ...). Configuration of
        # the link stays where the bring-up verified it: the environment.
        self.lib.check(
            L.ucp_config_read(None, None, ctypes.byref(config)), "ucp_config_read"
        )
        try:
            if self.net_devices is not None:
                # Pin THIS context to one link, instead of inheriting whatever
                # UCX_NET_DEVICES happens to be in the process environment.
                # The distinction matters on a host with more than one line:
                # the environment variable is a single process-wide value, so
                # it cannot express "this endpoint here, that one there",
                # while a config modified per context can. Left None the call
                # is skipped entirely and the config is exactly the one the
                # environment produced -- the unchanged default path.
                self.lib.check(
                    L.ucp_config_modify(
                        config, b"NET_DEVICES", self.net_devices.encode()
                    ),
                    f"ucp_config_modify(NET_DEVICES={self.net_devices})",
                )
            params = UcpParams()
            params.field_mask = UCP_PARAM_FIELD_FEATURES
            params.features = UCP_FEATURE_TAG
            ctx = ctypes.c_void_p()
            self.lib.check(
                L.ucp_init_version(
                    UCP_API_MAJOR, UCP_API_MINOR,
                    ctypes.byref(params), config, ctypes.byref(ctx),
                ),
                "ucp_init_version",
            )
        finally:
            L.ucp_config_release(config)
        self.ctx = ctx

        wparams = UcpWorkerParams()
        wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE
        # SINGLE is the fast mode. HTCCL collectives are issued from one
        # thread; the transport serialises with its own lock.
        wparams.thread_mode = UCS_THREAD_MODE_SINGLE
        worker = ctypes.c_void_p()
        self.lib.check(
            L.ucp_worker_create(self.ctx, ctypes.byref(wparams), ctypes.byref(worker)),
            "ucp_worker_create",
        )
        self.worker = worker
        self._eps: dict[int, ctypes.c_void_p] = {}
        self._closed = False
        # Request parameters are read synchronously by UCX during the
        # send/recv call and never retained, so one instance can back every
        # post. Building a fresh ctypes struct per transfer showed up as a
        # measurable slice of the per-collective overhead on a link whose
        # round trip is ~1.5 us.
        self._param = _make_request_param()
        self._param_ref = ctypes.byref(self._param)

        # Hot-path call targets bound once.
        #
        # A barrier on this link is ~12 us of which the wire is ~1.5 us; the
        # rest is Python. `self.lib.lib.ucp_tag_send_nbx` is three attribute
        # lookups and a bound-method frame per post, and the collectives issue
        # two posts plus several progress calls each. Binding the ctypes
        # function objects here removes that from every crossing. Measured
        # against a bare `ucp_worker_progress` of 0.26 us, the lookups are not
        # noise -- they are a comparable slice.
        L = self.lib.lib
        self._c_send = L.ucp_tag_send_nbx
        self._c_recv = L.ucp_tag_recv_nbx
        self._c_progress = L.ucp_worker_progress
        self._c_check = L.ucp_request_check_status
        self._c_free = L.ucp_request_free
        # An attribute rather than a method, deliberately: the pipelined
        # staging copies in the transport call this between sub-blocks, where
        # a Python frame would cost more than the call it wraps.
        self.progress = functools.partial(self._c_progress, self.worker)

    # -- address ------------------------------------------------------------

    def address(self) -> bytes:
        L = self.lib.lib
        addr = ctypes.c_void_p()
        length = ctypes.c_size_t()
        self.lib.check(
            L.ucp_worker_get_address(
                self.worker, ctypes.byref(addr), ctypes.byref(length)
            ),
            "ucp_worker_get_address",
        )
        try:
            return ctypes.string_at(addr, length.value)
        finally:
            L.ucp_worker_release_address(self.worker, addr)

    def connect(self, peer: int, address: bytes) -> None:
        """Create the persistent endpoint to ``peer``.

        A version-mismatched peer address is the failure this whole module is
        shaped around; the transport verifies versions *before* calling here,
        so a failure at this point means something else went wrong and the UCX
        status is reported verbatim.
        """
        buf = ctypes.create_string_buffer(address, len(address))
        params = UcpEpParams()
        params.field_mask = (
            UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE
        )
        params.address = ctypes.cast(buf, ctypes.c_void_p)
        # PEER error handling: a dead peer surfaces as a failed request rather
        # than an indefinite hang. Worth the small wireup cost for a link that
        # spans two physical machines.
        params.err_mode = UCP_ERR_HANDLING_MODE_PEER
        ep = ctypes.c_void_p()
        self.lib.check(
            self.lib.lib.ucp_ep_create(self.worker, ctypes.byref(params),
                                       ctypes.byref(ep)),
            f"ucp_ep_create(peer={peer})",
        )
        self._eps[peer] = ep

    # -- data path ----------------------------------------------------------

    # `progress` is bound in __init__.
    #
    # The two posts below are written flat -- no helper call, no eagerly
    # formatted description string -- because they are the per-collective hot
    # path. The old shape built `f"tag_send(peer={peer}, tag={tag:#x})"` on
    # EVERY post just to hand it to `_as_request`, which threw it away unless
    # the post failed; a hex format plus a string build per crossing is real
    # money at this scale. The message is now constructed only in the branch
    # that raises. `_as_request` is kept for callers outside the hot path.
    #
    # The pointer is passed as a plain int: `argtypes` already declares
    # `c_void_p`, so ctypes converts in C rather than us allocating a
    # `c_void_p` object per post.

    def post_send(self, peer: int, ptr: int, nbytes: int, tag: int):
        req = self._c_send(
            self._eps[peer], ptr, nbytes, tag, self._param_ref
        )
        if not req:
            return None  # immediate completion
        if req >= _ERR_PTR_FLOOR:
            raise UcxError(
                f"htccl ucx: tag_send(peer={peer}, tag={tag:#x}) failed: "
                f"{self.lib.status_string(_ptr_status(req))}"
            )
        return req

    def post_recv(self, ptr: int, nbytes: int, tag: int):
        req = self._c_recv(
            self.worker, ptr, nbytes, tag, _TAG_MASK_EXACT, self._param_ref
        )
        if not req:
            return None
        if req >= _ERR_PTR_FLOOR:
            raise UcxError(
                f"htccl ucx: tag_recv(tag={tag:#x}) failed: "
                f"{self.lib.status_string(_ptr_status(req))}"
            )
        return req

    def _as_request(self, req, what: str):
        """Normalise a ``ucs_status_ptr_t`` into None (done) or a request."""
        if not req:
            return None  # immediate completion
        if _ptr_is_err(req):
            raise UcxError(
                f"htccl ucx: {what} failed: "
                f"{self.lib.status_string(_ptr_status(req))}"
            )
        return req

    # How many spins pass between two clock reads in `wait`. The timeout it
    # guards is measured in minutes, so reading the clock on every pass buys
    # nothing and costs ~0.1 us of a ~0.7 us spin -- an eighth of the poll
    # granularity that decides how promptly a completion is noticed.
    _CLOCK_EVERY = 512

    def wait(self, requests: list) -> None:
        """Drive the worker until every outstanding request completes.

        Requests are polled as a set rather than one at a time so that a whole
        collective step's sends and receives make progress together; waiting on
        them sequentially would serialise exchanges that the wire can overlap,
        and can deadlock outright when two peers post their sends in the same
        order.
        """
        pending = [r for r in requests if r is not None]
        n = len(pending)
        if n == 0:
            return
        progress, check, free = self._c_progress, self._c_check, self._c_free
        worker = self.worker
        spins = 0
        deadline = time.monotonic() + self.timeout_s

        if n == 1:
            # The overwhelmingly common shape: a barrier, or a single-chunk
            # exchange whose send completed inline. Spelling it out avoids
            # rebuilding a one-element list on every pass of the spin -- which
            # at ~0.7 us per pass is a visible share of the poll granularity.
            req = pending[0]
            while True:
                progress(worker)
                status = check(req)
                if status != UCS_INPROGRESS:
                    free(req)
                    if status != UCS_OK:
                        raise UcxError(
                            f"htccl ucx: request failed: "
                            f"{self.lib.status_string(status)}"
                        )
                    return
                spins += 1
                if not spins % self._CLOCK_EVERY and time.monotonic() > deadline:
                    self._timeout(1)

        if n == 2:
            # The other common decode shape: one receive plus one send that
            # did not complete inline (a world-2 single-chunk exchange).
            # Two locals instead of a list rebuilt on every pass -- at ~0.7 us
            # per pass and a handful of passes per collective, the rebuild was
            # a measurable slice of the poll granularity.
            r0, r1 = pending
            while True:
                progress(worker)
                if r0 is not None:
                    status = check(r0)
                    if status != UCS_INPROGRESS:
                        free(r0)
                        if status != UCS_OK:
                            raise UcxError(
                                f"htccl ucx: request failed: "
                                f"{self.lib.status_string(status)}"
                            )
                        if r1 is None:
                            return
                        r0 = None
                if r1 is not None:
                    status = check(r1)
                    if status != UCS_INPROGRESS:
                        free(r1)
                        if status != UCS_OK:
                            raise UcxError(
                                f"htccl ucx: request failed: "
                                f"{self.lib.status_string(status)}"
                            )
                        if r0 is None:
                            return
                        r1 = None
                spins += 1
                if not spins % self._CLOCK_EVERY and time.monotonic() > deadline:
                    self._timeout((r0 is not None) + (r1 is not None))

        while pending:
            progress(worker)
            still = []
            for req in pending:
                status = check(req)
                if status == UCS_INPROGRESS:
                    still.append(req)
                    continue
                free(req)
                if status != UCS_OK:
                    raise UcxError(
                        f"htccl ucx: request failed: "
                        f"{self.lib.status_string(status)}"
                    )
            pending = still
            spins += 1
            if (
                pending
                and not spins % self._CLOCK_EVERY
                and time.monotonic() > deadline
            ):
                self._timeout(len(pending))

    def _timeout(self, n_pending: int) -> None:
        raise UcxError(
            f"htccl ucx: {n_pending} request(s) still pending after "
            f"{self.timeout_s:.0f}s. The peer is unreachable or the two "
            f"sides disagree about the collective sequence. Check "
            f"UCX_TLS / UCX_NET_DEVICES and that every rank issues the "
            f"same collectives in the same order."
        )

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        L = self.lib.lib
        param = UcpRequestParam()
        param.op_attr_mask = 0
        param.flags = UCP_EP_CLOSE_FLAG_FORCE
        for ep in self._eps.values():
            try:
                req = L.ucp_ep_close_nbx(ep, ctypes.byref(param))
                if req and not _ptr_is_err(req):
                    deadline = time.monotonic() + 5.0
                    while L.ucp_request_check_status(req) == UCS_INPROGRESS:
                        L.ucp_worker_progress(self.worker)
                        if time.monotonic() > deadline:
                            break
                    L.ucp_request_free(req)
            except Exception:
                pass  # teardown must never mask the real error
        self._eps.clear()
        try:
            L.ucp_worker_destroy(self.worker)
        finally:
            L.ucp_cleanup(self.ctx)
