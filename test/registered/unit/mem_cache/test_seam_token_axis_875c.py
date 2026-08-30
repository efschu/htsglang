"""#875 axis 3: the token axis, answered at the code -- and the verdict is DO NOT BUILD.

`seam_layer_carry.py` settled the LAYER axis and named two more. The head axis was
settled (replicated in both phases, no remap). This file settles the third, which
I named as the gate on the whole carry, and it settles it against the source
rather than against a model in my head.

Q1 -- DOES `req_to_token` IN THE TP PHASE HOLD ALL TOKENS OR ONLY THE OWNED ONES?
ALL of them, and the values are GLOBAL cache slots. `dcp_weighted_read_slots`
(layers/dcp/owner.py:440-465) says so in its own signature: it takes "a flat list
of GLOBAL cache slots (typically the ``req_to_token`` rows of a paged read)" and
RETURNS `(compact, owned)`. The table is one row per request of width
`max_context_len` (memory_pool.py:373-375) -- global context positions, allocated
identically on every rank. The owner rule is applied DOWNSTREAM, when the
attention backend builds its kv_indices; it is not baked into the table.

So the owner rule at owner.py:159 describes the ASSIGNMENT, exactly as the
briefing suspected, and says nothing about what the table materialises. The table
materialises the global list.

Q2 -- HOW DOES THE SOURCE'S POSITIONAL ROW LIST MAP ONTO THE DESTINATION'S
COMPACTED ROWS? IT DOES NOT. There is no such function, and that absence is the
finding. The only global->compact mapping in the tree is
`dcp_weighted_read_slots` / `dcp_weighted_write_slots`, and its ONLY callers are
the attention backends (`layers/attention/triton_backend.py:2392` and the
flashinfer twin). Nothing on the seam / offload path calls either. Verified
below by an AST sweep of the whole `srt/` tree, not by reading one file.

The sweep had to be AST and not text, and it was caught being text:
`managers/prefetch_ballot.py` NAMES `dcp_weighted_read_slots` in a docstring
about an unrelated extraction, and a substring match reported it as a caller.
That is the third time on this branch that a name in prose has been mistaken for
a use (the 4 KiB threshold, `outgoing_bytes=0` in its own docstring, and now
this), so the sweep carries its own control below.

AND THAT HAS A CONSEQUENCE BIGGER THAN THE CARRY. `Req.load_kv_cache`
(schedule_batch.py:1806) hands `req_to_token[req_pool_idx, :seqlen-1]` -- global
slots -- straight to `load_cpu_copy`, which indexes physical buffer rows with
them. `HybridLinearKVPool` (this rig's pool) forwards `indices` untouched; it
translates only the MAMBA ids, and says so in its own comment. Under the PP phase
`dcp_size == 1`, global == physical, and the path is correct. Under the TP phase
with a weighted vector they diverge. Whether that is a LIVE defect depends on
whether this path runs in the TP phase with uneven DCP active, which is a boot
question and is NOT claimed here. It is recorded as the next check, with its
address.

Note the contrast, because it shows the tree already knows this shape:
`UnifiedSWAKVPool.get_cpu_copy` (unified_memory_pool.py:1232) DOES translate,
via `_virt_tokens_to_phys_tokens`, before touching a sub-pool. One pool family
translates on this path and the other does not.

Q3 -- CAN THE LAYER GATHER AND THE TOKEN SCATTER FOLD INTO ONE COLLECTIVE?
YES, and it is an all-to-all. Rank r holds (its layers x ALL tokens). Rank r'
needs (ALL layers x its owned tokens). So r must send r' exactly the block
(L_r x T_r') -- a distinct block per peer, which is the definition of all-to-all.
Not a gather followed by a scatter: one collective.

THE VERDICT, which the stop criterion explicitly permits: DO NOT BUILD.

  payload   = n_layers x extent x 2 (K,V) x kv_heads x head_dim x dtype_bytes
  specimen  = 16 layers x 13 rows, fp8 KV. At a typical 8x128 GQA head config
              that is ~2 KiB per layer-token, i.e. ~416 KiB for the whole
              request across the whole group.
  transport = NOT PRICED. WITHDRAWN, see below.
  refusal   = recompute `extent` tokens of prefill. For 13 tokens that is
              sub-millisecond of GPU work.

THE TRANSPORT TERM IS WITHDRAWN, and the citation sweep is why. I attributed
"43.9 KiB per crossing, 166 us enqueue, 1777-9201 us receive" to #656. It is not
in #656. It is in `13c55d7b86` "[PP] #201 slice 2: the stage boundary across two
rigs" -- a CROSS-RIG, two-node measurement over a 40G line, on a different model
(Qwen3.5-4B fp16), timing PP microbatch crossings rather than an intra-node
collective. Worse, its own text refuses the reading I gave it: "`recv` is
BLOCKING, i.e. bubble plus wire -- 9.2 ms on stage 1 is that stage waiting for
stage 0, NOT the 40G line". I used a pipeline bubble as a transport latency, from
the wrong ticket, for the wrong link. I have no measured local collective figure
and am not substituting one.

SO THE VERDICT NO LONGER RESTS ON A TIMING COMPARISON. What carries it instead,
neither term needing a transport number:

  1. THE HEAD AXIS IS LOSSY. PP holds 4 kv-heads per layer and TP holds 1
     (`max(1, 4 // attn_tp_size)`). PP->TP is not a remap at all until someone
     decides WHICH heads survive, and no such rule exists. A carry cannot be
     built over an undefined reduction.
  2. A new collective in the cutover's NO-RETURN region is the #630 wedge shape.
     That is a risk argument, unbounded, and independent of how fast the link is.

Point 1 alone is decisive. The payload being sub-MiB at specimen scale still
holds and still means any collective would be latency-dominated -- but that is
now a remark, not the load-bearing term.

WHAT WOULD CHANGE THE VERDICT, stated so it is falsifiable rather than final:
`extent` is `seqlen - 1`, the request's whole context, so it is 13 only because
the specimen's request was short. The collective's cost is roughly
extent-independent at these sizes while the recompute is linear in it, so there
is a break-even, and it is somewhere in the hundreds-to-thousands of tokens. The
measurement that would settle it is the DISTRIBUTION of `extent` over requests
actually retracted at a flip. I do not have it, and I am not entitled to infer
it from any record. Until it exists, the refusal is the cheaper of the two wrong
answers and the carry stays unbuilt.

Hermetic: source inspection and arithmetic. No CUDA, no collectives, no device.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import inspect
import unittest
from pathlib import Path

from sglang.test.test_utils import CustomTestCase


class TestQ1TheTableHoldsGlobalSlots(CustomTestCase):
    """Answered at the code, with the docstring that states it."""

    def test_the_read_rule_declares_its_input_is_the_req_to_token_row(self):
        from sglang.srt.layers.dcp.owner import dcp_weighted_read_slots

        doc = dcp_weighted_read_slots.__doc__ or ""
        self.assertIn("GLOBAL cache slots", doc)
        self.assertIn("req_to_token", doc)

    def test_the_read_rule_RETURNS_the_compaction_rather_than_assuming_it(self):
        """`(compact, owned)` out of global in -- i.e. the compaction happens at
        the CALLER, downstream of the table, not inside it."""
        from sglang.srt.layers.dcp.owner import dcp_weighted_read_slots

        src = inspect.getsource(dcp_weighted_read_slots)
        self.assertIn("return compact, owned", src)

    def test_the_table_is_allocated_per_global_context_position(self):
        """One row per request of width `max_context_len` -- global positions,
        identical on every rank. Not a per-rank owned subset."""
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        src = inspect.getsource(ReqToTokenPool.__init__)
        self.assertIn("(self._alloc_size, max_context_len)", src)


class TestQ2NoMappingExistsOnTheSeamPath(CustomTestCase):
    """The absence IS the finding, so it is asserted as an absence -- by sweep,
    so that a mapping added later flips this test rather than going unnoticed."""

    SEAM_PATH_FILES = (
        "managers/schedule_batch.py",
        "mem_cache/memory_pool.py",
        "mem_cache/allocator/base.py",
        "mem_cache/allocator/paged.py",
        "mem_cache/allocator/token.py",
        "mem_cache/allocator/swa.py",
    )

    def _srt(self):
        import sglang.srt.mem_cache.memory_pool as mp

        return Path(mp.__file__).parent.parent

    RULE_NAMES = {"dcp_weighted_read_slots", "dcp_weighted_write_slots"}

    def _rule_users(self):
        """Files that IMPORT or CALL the owner rule -- by AST, not by substring.

        A substring sweep is wrong here and was caught being wrong:
        `managers/prefetch_ballot.py` names `dcp_weighted_read_slots` in PROSE,
        in a docstring about an unrelated extraction, and a text match read that
        as a caller. Same shape as this branch's docstring-versus-call miss on
        `outgoing_bytes=0`; a name in prose is not a use."""
        root = self._srt()
        users = []
        for path in sorted(root.rglob("*.py")):
            rel = str(path.relative_to(root))
            if rel.startswith("layers/dcp/"):
                continue  # the definition and its own re-exports
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:  # pragma: no cover
                continue
            hit = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if any(a.name in self.RULE_NAMES for a in node.names):
                        hit = True
                elif isinstance(node, ast.Name) and node.id in self.RULE_NAMES:
                    hit = True
                elif isinstance(node, ast.Attribute) and node.attr in self.RULE_NAMES:
                    hit = True
            if hit:
                users.append(rel)
        return users

    def test_the_owner_rule_is_used_only_by_attention_backends(self):
        users = self._rule_users()
        self.assertTrue(users, "the sweep found nothing -- it is broken, not the tree")
        for rel in users:
            self.assertTrue(
                rel.startswith("layers/attention/"),
                f"{rel} now USES the owner rule. If the seam path gained a "
                f"global->compact translation, axis 3 has moved and this file's "
                f"verdict must be redone.",
            )

    def test_the_sweep_distinguishes_a_call_from_a_mention(self):
        """CONTROL for the sweep itself. A file that merely NAMES the rule in a
        docstring must not be reported as a user, or the test above is a text
        grep wearing an AST costume."""
        import sglang.srt.managers.prefetch_ballot as pb

        self.assertIn("dcp_weighted_read_slots", Path(pb.__file__).read_text())
        self.assertNotIn("managers/prefetch_ballot.py", self._rule_users())

    def test_no_seam_path_file_translates_global_to_compact(self):
        root = self._srt()
        for rel in self.SEAM_PATH_FILES:
            src = (root / rel).read_text()
            self.assertNotIn("dcp_weighted_read_slots", src, f"{rel} translates now")
            self.assertNotIn("dcp_weighted_write_slots", src, f"{rel} translates now")

    def test_the_rig_pool_forwards_kv_indices_untranslated(self):
        """`HybridLinearKVPool.get_cpu_copy` translates the MAMBA ids and passes
        the KV indices through. Asserted on the AST so a comment mentioning
        translation cannot satisfy it."""
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        import textwrap

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(HybridLinearKVPool.get_cpu_copy))
        )
        translated = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "get_cpu_copy" and node.args:
                    translated.add(ast.dump(node.args[0]))
        plain = {d for d in translated if "_mamba_translate" not in d}
        self.assertTrue(
            plain, "expected at least one untranslated forward of the KV indices"
        )

    def test_the_unified_pool_DOES_translate_which_is_the_contrast(self):
        """One pool family translates on this path and the other does not. If
        this ever stops being true the asymmetry is gone and so is the finding."""
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool

        src = inspect.getsource(UnifiedSWAKVPool.get_cpu_copy)
        self.assertIn("_virt_tokens_to_phys_tokens", src)


class TestQ3OneCollectiveNotTwo(CustomTestCase):
    """The exchange shape, as arithmetic over the two geometries."""

    N_LAYERS = 16
    PP_STAGES = ((0, 8), (8, 4), (12, 4))  # (start_layer, layer_num)
    TOKEN_VECTOR = (29, 19, 16)  # uneven_token_vector from the boot args

    def _owned(self, extent, rank):
        """Which of `extent` global positions rank `rank` owns, by the weighted
        owner rule's block structure."""
        S = sum(self.TOKEN_VECTOR)
        lo = sum(self.TOKEN_VECTOR[:rank])
        hi = lo + self.TOKEN_VECTOR[rank]
        return [t for t in range(extent) if lo <= (t % S) < hi]

    def test_every_rank_needs_a_DISTINCT_block_from_every_peer(self):
        """Distinct block per (sender, receiver) pair is exactly all-to-all. A
        gather-then-scatter would be two collectives; this is one."""
        extent = 200
        blocks = {}
        for s, (start, n) in enumerate(self.PP_STAGES):
            for r in range(3):
                blocks[(s, r)] = (
                    frozenset(range(start, start + n)),
                    tuple(self._owned(extent, r)),
                )
        self.assertEqual(9, len(blocks))
        distinct = {v for v in blocks.values()}
        self.assertEqual(
            9,
            len(distinct),
            "some (sender, receiver) blocks coincide -- then it "
            "is not an all-to-all and the cost model changes",
        )

    def test_the_union_of_what_a_rank_receives_is_exactly_what_it_needs(self):
        """Correctness of the shape: all layers, own tokens, no gap, no overlap."""
        extent = 200
        for r in range(3):
            layers = set()
            for start, n in self.PP_STAGES:
                layers |= set(range(start, start + n))
            self.assertEqual(set(range(self.N_LAYERS)), layers)
            owned = self._owned(extent, r)
            self.assertEqual(len(owned), len(set(owned)))

    def test_the_token_vector_partitions_every_position_exactly_once(self):
        """No position is owned twice and none is orphaned -- otherwise the
        all-to-all would drop or duplicate KV."""
        extent = 500
        seen = []
        for r in range(3):
            seen.extend(self._owned(extent, r))
        self.assertEqual(sorted(seen), list(range(extent)))


class TestTheVerdictArithmetic(CustomTestCase):
    """The pricing, so the verdict is falsifiable rather than asserted."""

    def test_the_specimen_payload_is_small_enough_to_be_latency_bound(self):
        extent, layers, per_layer_token_bytes = 13, 16, 2048
        payload = layers * extent * per_layer_token_bytes
        self.assertLess(
            payload,
            1 << 20,
            "the specimen's whole-group payload is under a MiB, so a collective "
            "for it is latency-bound and the verdict stands",
        )

    def test_the_break_even_is_governed_by_extent_not_by_the_specimen(self):
        """The verdict is conditional and this says on what. Payload is linear
        in extent, so a long-context retraction is a different question."""
        per_layer_token_bytes, layers = 2048, 16
        small = layers * 13 * per_layer_token_bytes
        large = layers * 100_000 * per_layer_token_bytes
        self.assertGreater(large // small, 1000)


if __name__ == "__main__":
    unittest.main()
