"""The retired second HiCache implementation is refused, not merely unused.

User order 2026-08-29: "das muss sofort allem ausgetrieben werden und ueberall
warnings dazu, dass das niemals wieder passiert." Two seams carry that order,
and each gets a can-fail proof here rather than a comment:

1. STORE ATTACH. The era's per-stage component writer keys a GDN blob
   ``{hash}.mamba{model}_{identity}_{tp}_{tp_size}_{pp_size}_{pp_rank}`` -- the
   ``_0_1_3_r`` tail -- while the #706 writer that replaced it keys the same
   pool geometry-neutral. Both ran against one directory for weeks. Measured
   2026-08-29 in the specimen store ``/tmp/hicache_783``: 328 canonical
   ``.mamba`` blobs beside 1091 per-stage ones. That is one content-addressed
   hash with two different cuts of the state behind it, and the read path
   cannot tell them apart. ``MixedGenerationError`` refuses it at attach, on
   #558's own mechanic (it subclasses ``MixedLayoutError``) one axis over:
   #558 is the FILE-LAYOUT axis, this is the KEY-FORMAT axis.

2. CONSTRUCTION OF THE RETIRED READ PATH. ``HiMambaRadixCache`` is the era's
   own storage read path -- its own prefetch, its own host-pool attach -- with
   no construction site since #581. It now refuses to be built at all, so a
   future call site cannot silently restore two live HiCache implementations
   against one store.

WHAT THESE TESTS DO NOT CLAIM. The generation audit is a bounded detector: it
stops early, and a clean result bounds the check rather than proving the store
coherent. ``test_the_audit_reports_its_own_bound`` pins that honesty so a later
reader does not upgrade a clean audit into a guarantee.
"""

import os
import tempfile
import unittest

from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageSpec
from sglang.srt.mem_cache.canonical_page_store import window_for_layers
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    MixedGenerationError,
    MixedLayoutError,
    audit_blob_generations,
    page_shard,
)
from sglang.test.test_utils import CustomTestCase

ATTN_LAYER_IDS = list(range(3, 64, 4))
CELL = 64
SPEC = CanonicalPageSpec(
    num_attn_layers=len(ATTN_LAYER_IDS), kv_bytes_per_token_per_attn_layer=CELL
)
IDENTITY = "0123456789abcdef"
MODEL = "Qwen3.6-27B"

# The two suffixes HiCacheFile derives for this config. Spelled out here on
# purpose: if the suffix rule ever changes, this test must break loudly rather
# than keep planting files no gate looks at.
STAGE_SUFFIX = f"_{MODEL}_{IDENTITY}_0_1_3_0"
CANONICAL_SUFFIX = f"_{MODEL}_{IDENTITY}"


def _window():
    return window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS[:7])


def _backend(root, *, window=None):
    return HiCacheFile(
        HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=3,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name=MODEL,
            model_identity_hash=IDENTITY,
            canonical_kv_page=window,
        ),
        file_path=root,
    )


def _plant(root, stem):
    """Write an empty page file at the sharded path ``stem`` would occupy."""
    shard = os.path.join(root, page_shard(stem))
    os.makedirs(shard, exist_ok=True)
    path = os.path.join(shard, f"{stem}.bin")
    with open(path, "wb") as f:
        f.write(b"\0" * 8)
    return path


class TestStoreRefusesTwoBlobGenerations(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_both_generations_in_one_store_is_refused_at_attach(self):
        """THE CAN-FAIL PROOF for the store seam: this is the specimen shape of
        /tmp/hicache_783, reduced to two files."""
        _plant(self.root, f"cafe01.mamba{STAGE_SUFFIX}")
        _plant(self.root, f"cafe02.mamba{CANONICAL_SUFFIX}")
        with self.assertRaises(MixedGenerationError) as cm:
            _backend(self.root, window=_window())
        msg = str(cm.exception)
        self.assertIn("TWO key generations", msg)
        self.assertIn("retired second HiCache implementation", msg)
        # The refusal must hand the reader the way out, not just the door.
        self.assertIn("#975", msg)

    def test_the_refusal_is_a_mixed_layout_error(self):
        """#558's mechanic is REUSED, not copied: an existing handler for the
        layout refusal keeps catching the generation refusal."""
        self.assertTrue(issubclass(MixedGenerationError, MixedLayoutError))

    def test_only_the_retired_generation_present_warns_and_attaches(self):
        """Not yet ambiguous, so not yet a refusal -- but it must be named,
        because the next canonical write turns it into one."""
        _plant(self.root, f"cafe01.mamba{STAGE_SUFFIX}")
        with self.assertLogs(
            "sglang.srt.mem_cache.hicache_storage", level="WARNING"
        ) as logs:
            _backend(self.root, window=_window())
        self.assertTrue(
            any("retired second HiCache" in line for line in logs.output), logs.output
        )

    def test_only_the_canonical_generation_present_attaches_silently(self):
        """The gate must not fire on the state we are migrating TOWARDS."""
        _plant(self.root, f"cafe02.mamba{CANONICAL_SUFFIX}")
        _backend(self.root, window=_window())  # no raise

    def test_draft_blobs_are_not_a_second_generation(self):
        """Draft KV keeps the geometry suffix BY DESIGN in every generation
        (``_is_shared_kv_key``'s docstring). The specimen store is full of
        ``.draft-*_0_1_3_r`` files; flagging those would refuse every correct
        store with speculative decoding on."""
        _plant(self.root, f"cafe03.draft-abc{STAGE_SUFFIX}")
        _plant(self.root, f"cafe02.mamba{CANONICAL_SUFFIX}")
        _backend(self.root, window=_window())  # no raise

    def test_the_old_world_alone_is_not_audited(self):
        """Without the canonical KV page the two suffixes are identical, so
        there is nothing to disambiguate and no audit to pay for."""
        _plant(self.root, f"cafe01.mamba{STAGE_SUFFIX}")
        backend = _backend(self.root, window=None)  # no raise
        self.assertEqual(backend.config_suffix, backend.kv_config_suffix)

    def test_the_audit_reports_its_own_bound(self):
        """A clean audit that stopped early must say so, so a reader cannot
        upgrade 'found nothing' into 'the store is coherent'."""
        for i in range(6):
            _plant(self.root, f"ca{i:02d}ff.mamba{CANONICAL_SUFFIX}")
        stage, canonical, seen, exhausted = audit_blob_generations(
            self.root,
            stage_marker=f".mamba{STAGE_SUFFIX}",
            canonical_marker=f".mamba{CANONICAL_SUFFIX}",
            max_files=2,
        )
        self.assertEqual(stage, ())
        self.assertFalse(exhausted)
        self.assertLessEqual(seen, 3)

        stage, canonical, seen, exhausted = audit_blob_generations(
            self.root,
            stage_marker=f".mamba{STAGE_SUFFIX}",
            canonical_marker=f".mamba{CANONICAL_SUFFIX}",
        )
        self.assertTrue(exhausted)
        self.assertEqual(seen, 6)

    def test_a_missing_store_is_not_an_error(self):
        stage, canonical, seen, exhausted = audit_blob_generations(
            os.path.join(self.root, "does-not-exist"),
            stage_marker=".mamba_x",
            canonical_marker=".mamba_y",
        )
        self.assertEqual((stage, canonical, seen), ((), (), 0))


class TestRetiredReadPathRefusesConstruction(CustomTestCase):
    def test_importing_the_retired_module_warns(self):
        """Seam (a): the import is how the module gets back into a process."""
        import importlib

        import sglang.srt.mem_cache.hi_mamba_radix_cache as mod

        with self.assertLogs(
            "sglang.srt.mem_cache.hi_mamba_radix_cache", level="WARNING"
        ) as logs:
            importlib.reload(mod)
        self.assertTrue(
            any("RETIRED MODULE" in line for line in logs.output), logs.output
        )

    def test_constructing_the_retired_cache_refuses_by_name(self):
        """THE CAN-FAIL PROOF for the class seam. Built with a deliberately
        empty params object: the refusal must come BEFORE any argument
        validation, or a future caller with valid pools would slip past it."""
        from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

        with self.assertRaises(NotImplementedError) as cm:
            HiMambaRadixCache(object(), object())
        msg = str(cm.exception)
        self.assertIn("retired second HiCache implementation", msg)
        self.assertIn("hi_mamba_radix_cache", msg)
        # Names the replacement, per the order's "Verweis auf den Neu-Pfad".
        self.assertIn("UnifiedRadixCache", msg)

    def test_the_registry_still_has_no_construction_site(self):
        """Belt and braces with the refusal: a call site added anywhere would
        now raise at runtime, but this catches it at desk time."""
        import subprocess

        root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python")
        out = subprocess.run(
            ["grep", "-rn", "HiMambaRadixCache(", os.path.abspath(root)],
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        sites = [ln for ln in out if "class HiMambaRadixCache(" not in ln]
        self.assertEqual(sites, [], sites)


class TestEraAdmissionRingPredicate(CustomTestCase):
    """strip-B (d4478a053a) gates the era's admission-decision ring off on
    plain PP. Its own evidence line reads "py_compile" -- no suite ever saw
    it, and cherry-picked onto the #1010 pin it turned 15 green tests into 13
    failures (`'types.SimpleNamespace' object has no attribute 'server_args'`)
    across the #787/#791/#795/#796 stand-in family. This is the can-fail proof
    that predicate never had."""

    def _predicate(self):
        from sglang.srt.managers.scheduler_pp_mixin import _pp_era_ring_live

        return _pp_era_ring_live

    def test_flip_off_strips_the_era_ring(self):
        import types

        holder = types.SimpleNamespace(
            server_args=types.SimpleNamespace(enable_phase_flip=False)
        )
        self.assertFalse(self._predicate()(holder))

    def test_flip_on_keeps_the_era_ring(self):
        """Briefing constraint: the flip mechanism itself is NOT retired and
        must keep working. The discriminator boot flip-ON depends on this arm."""
        import types

        holder = types.SimpleNamespace(
            server_args=types.SimpleNamespace(enable_phase_flip=True)
        )
        self.assertTrue(self._predicate()(holder))

    def test_a_stand_in_without_server_args_keeps_the_pre_gate_behaviour(self):
        """THE REGRESSION strip-B shipped. A holder that never set server_args
        must neither raise nor be read as 'flip off'."""
        import types

        self.assertTrue(self._predicate()(types.SimpleNamespace()))

    def test_the_hop_is_unconditional_on_both_ends(self):
        """#1014. THE CONTRACT INVERTED, AND WITH IT WHAT THIS TEST PINS.

        #1013 asserted the opposite of this: that the era predicate GUARDS the
        admission-decision receive, so sender and receiver are gated in step.
        That shape was congruent and served nothing. The reason is a fact
        neither #1013 nor the strip-B gates it copied had measured -- the hop
        is not the era's ring, it is a MULTIPLEX CARRIER. Six facts from five
        tickets ride the one message (OUTPUT_EXPECTED #791b, PASS_VOIDED and
        UPSTREAM_LAUNCHED #797, SLOT_OCCUPANT #1000b, LAUNCHED_CHAIN #978,
        PARKED_CONTINUATION #968); six of the module's seven ``_PP_*_KEY`` are
        on it, only ``_PP_VOID_OUTPUT_KEY`` rides the output wrap. Switching
        the hop off retires five foreign wires along with the era's own.

        Measured, boot boot_pp3solo_0d81aab8db_0829_095245.log: the first of
        the six to be missed, ``_pp_upstream_launched_incoming`` (one writer,
        behind the skipped read), left ``_pp_drain_voided_proxy`` declining
        every drain -- "voided proxy drained" 0 times in the whole log -- and
        PP0 parked for ever on ``pp-ring-commit/dict/send_proxy_work[0]``.
        0 prefill batches, 0 http=200.

        So the era payload is emptied and the MESSAGE keeps flying. Congruence
        stops being a property two gates must maintain and becomes one of the
        construction: both ends unconditional, nothing to keep in step. This
        pins that shape, and it is the direct inverse of what #1013 pinned --
        which is the point, and why the old assertion is replaced rather than
        extended.
        """
        import inspect

        from sglang.srt.managers import scheduler_pp_mixin as mod

        # COUNTED FROM THE AST, NOT FROM THE TEXT, and that is the whole
        # lesson of this test's two previous versions. #1013's tally counted
        # one SPELLING and missed a gate written in another. The first draft
        # of THIS assertion counted both spellings as substrings and promptly
        # matched a PROSE mention of `_pp_era_ring_live(self)` inside a
        # comment at :4131, reporting two call sites where the module has one.
        # A substring count cannot tell a call from a sentence about a call;
        # the parser can, so the parser does the counting.
        import ast

        tree = ast.parse(inspect.getsource(mod))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_pp_era_ring_live"
        ]
        enclosing = []
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for n in ast.walk(fn):
                    if n in calls:
                        enclosing.append(fn.name)
        self.assertEqual(
            len(calls),
            1,
            "the era predicate must survive at exactly ONE call site -- the "
            "payload choice in `_pp_send_admission_decision`. A second site "
            "means something is being switched off again rather than "
            f"emptied; found {len(calls)} in {sorted(set(enclosing))}",
        )
        self.assertEqual(sorted(set(enclosing)), ["_pp_send_admission_decision"])

        # The receive still happens, and it runs every pass. "No era
        # predicate in the body" needs no separate text assertion -- the AST
        # count above already proves the module's only call lives in
        # `_pp_send_admission_decision`, so any gate here would have failed
        # it. A substring check here would only re-run the same prose trap:
        # the #1014 comment in this very body QUOTES the removed gate.
        body = inspect.getsource(mod.SchedulerPPMixin._event_loop_pp_body)
        self.assertIn("_pp_recv_admission_decision(", body)

        # The send runs every pass too, and the surviving predicate chooses
        # the PAYLOAD rather than whether to send.
        send = inspect.getsource(mod.SchedulerPPMixin._pp_send_admission_decision)
        self.assertIn("_pp_era_ring_live(self)", send)
        self.assertIn(
            "entries=()",
            send,
            "with the era off the decision must be sent EMPTY, not skipped.",
        )
        gate_line = send[send.index("_pp_era_ring_live(self)") :].splitlines()[1]
        self.assertNotIn(
            "return",
            gate_line,
            "the era predicate must select an empty payload, never return "
            f"early: {gate_line.strip()!r}",
        )

        # The reap is unconditional, or every pass leaks an un-waited isend.
        reap = inspect.getsource(mod.SchedulerPPMixin._pp_commit_admission_send_work)
        self.assertNotIn("_pp_era_ring_live(self)", reap)

    def test_the_output_return_path_keeps_its_deadline(self):
        """The bound on `_pp_commit_comm_work` is an INSTRUMENT on a core PP
        channel, not retired machinery, and must not be gated off with the era.

        Operator boot boot_pp3solo_769f88efea_0829_092829.log, PP3 solo with NO
        flip: all three ranks wedged on the first request at this commit, on
        'pp-ring-commit/dict/send_output_work[0]' and 'pp-ring-commit/p2p[0]'.
        The channel is the output return path (last stage -> PP0), which plain
        upstream PP has. Silencing the deadline there turns a named 120 s death
        into a park against gloo's two-hour timeout."""
        import inspect

        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        src = inspect.getsource(SchedulerPPMixin._pp_commit_comm_work)
        # The predicate itself, not the words about it: the comment above the
        # code deliberately quotes strip-B's `budget = 0.0` line.
        self.assertNotIn("if not _pp_era_ring_live(self):", src)
        self.assertIn("bounded_wait(", src)


if __name__ == "__main__":
    unittest.main()
