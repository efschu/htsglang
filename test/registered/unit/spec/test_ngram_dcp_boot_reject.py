"""NGRAM speculation x DCP must be refused at BOOT, not at the first verify.

THE DEFECT
----------
``NGRAM_VERIFY`` is deliberately not a member of either backend's
``_DCP_VERIFY_SPEC_INPUT_TYPES`` (#180.3: four SpecInputTypes end in
``_VERIFY``, and the M4 split was written for the two that present a uniform
draft-token query block over a LINEAR draft chain). The refusal is loud, so
there is no correctness risk -- but it fires in the target-verify METADATA
BUILD, i.e. at the first verify step of the first request. Everything before
that succeeds: weights load, pools are sized, graphs are captured, the server
reports ready. The user pays the whole model load to learn about a
configuration error that is decidable from ``server_args`` alone.

Both facts (this is the ngram algorithm; this is the DCP size) are already
known at argument resolution, so the primary refusal lives in
``ServerArgs._handle_dcp_validation`` (#229, covered by
``test/registered/unit/server_args/test_spec_dcp_boot_reject.py``) and fires
before any weights load. ``NGRAMWorker.__init__`` repeats the call as a
backstop for directly constructed workers, named the way its siblings are:
``reject_unsupported_dcp_geometry`` in the Triton backend and
``reject_silently_inert_dcp`` in the flashinfer one. The gate function itself
lives in ``spec_info``, next to the ``SpecInputType`` membership it reasons
about.

WHY ``dcp_size > 1`` AND NOT ``uneven_dcp``
-------------------------------------------
Both backends fail, through different doors, and neither door needs the
weighted owner rule:

* Triton refuses explicitly -- ``init_forward_metadata``'s target-verify arm
  gates on ``self.dcp_size > 1`` and raises ``NotImplementedError`` for any
  spec input type outside the set, EVEN and uneven DCP alike.
* flashinfer's verify split is an ``elif`` on ``uneven_dcp and spec type in
  the set``; an ngram verify misses it and drops into the generic spec branch,
  which leaves ``use_ragged`` False while ``_forward_extend_dcp`` runs the
  ragged stage unconditionally -> ``AttributeError`` (the same shape the
  DFLASH verify hit before it joined the set). Its EVEN-DCP case cannot arise:
  ``reject_silently_inert_dcp`` already refuses a ``--dcp-size`` that the
  backend would ignore.

So ``dcp_size > 1`` is the exact reachable condition, not a widening of it.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestNgramDcpBootReject(unittest.TestCase):
    def test_the_gate_refuses_every_dcp_size_above_one(self):
        from sglang.srt.speculative.ngram_worker import reject_ngram_verify_under_dcp

        for dcp_size in (2, 3, 4, 8):
            with self.subTest(dcp_size=dcp_size):
                with self.assertRaises(ValueError) as cm:
                    reject_ngram_verify_under_dcp(dcp_size)
                msg = str(cm.exception)
                # The message must name the condition, not just fail.
                self.assertIn("NGRAM_VERIFY", msg)
                self.assertIn(f"--dcp-size {dcp_size}", msg)

    def test_dcp_size_one_is_inert(self):
        """The default path must not move. dcp_size <= 1 is every stock boot."""
        from sglang.srt.speculative.ngram_worker import reject_ngram_verify_under_dcp

        for dcp_size in (0, 1):
            with self.subTest(dcp_size=dcp_size):
                reject_ngram_verify_under_dcp(dcp_size)  # must not raise

    def test_the_boot_path_actually_calls_the_gate(self):
        """A gate nobody calls is the bug one layer up (#182's lesson).

        ``NGRAMWorker.__init__`` is the boot site; check the call is there and
        that it precedes the corpus construction, so the refusal cannot land
        behind a several-GiB allocation.
        """
        import inspect

        from sglang.srt.speculative import ngram_worker

        src = inspect.getsource(ngram_worker.NGRAMWorker.__init__)
        self.assertIn("reject_ngram_verify_under_dcp(", src)
        self.assertLess(
            src.index("reject_ngram_verify_under_dcp("),
            src.index("NgramCorpus("),
            "the DCP refusal must fire before the corpus is allocated",
        )

    def test_the_frozen_worker_backstop_calls_its_gate(self):
        """Same shape for FROZEN_KV_MTP (#229): primary gate in ServerArgs,
        backstop as the first statement of the worker orchestrator's
        ``__init__``, before the draft worker is built."""
        import inspect

        from sglang.srt.speculative import frozen_kv_mtp_worker_v2

        src = inspect.getsource(
            frozen_kv_mtp_worker_v2.FrozenKVMTPWorkerV2.__init__
        )
        self.assertIn("reject_frozen_kv_mtp_verify_under_dcp(", src)
        self.assertLess(
            src.index("reject_frozen_kv_mtp_verify_under_dcp("),
            src.index("FrozenKVMTPDraftWorker("),
            "the DCP refusal must fire before the draft worker is built",
        )

    def test_the_late_detection_this_replaces_is_still_the_backstop(self):
        """Pin WHY the gate exists, so it cannot be deleted as redundant.

        NGRAM_VERIFY is absent from both backends' verify-split sets, and the
        Triton target-verify arm raises rather than falling through. Those are
        the runtime refusals; this gate only moves the moment of truth to boot,
        it does not replace them.
        """
        import sglang.srt.layers.attention.flashinfer_backend as fb
        import sglang.srt.layers.attention.triton_backend as tb
        from sglang.srt.speculative.spec_info import SpecInputType

        self.assertNotIn(SpecInputType.NGRAM_VERIFY, tb._DCP_VERIFY_SPEC_INPUT_TYPES)
        self.assertNotIn(SpecInputType.NGRAM_VERIFY, fb._DCP_VERIFY_SPEC_INPUT_TYPES)

    def test_the_verify_type_coverage_audit(self):
        """Every ``*_VERIFY`` type is served by the split or gated at boot.

        Four types end in ``_VERIFY``. Two (EAGLE, DFLASH) are served. NGRAM is
        gated by ``reject_ngram_verify_under_dcp`` and FROZEN_KV_MTP by
        ``reject_frozen_kv_mtp_verify_under_dcp`` (#229 closed the last
        late-detection hole; both gates fire in
        ``ServerArgs._handle_dcp_validation``). If a fifth verify layout is
        added, this test is where its DCP story gets recorded: served by the
        split, or gated at boot.
        """
        import sglang.srt.layers.attention.triton_backend as tb
        from sglang.srt.speculative.spec_info import SpecInputType

        verify_types = {
            t for t in SpecInputType if t.name.endswith("_VERIFY")
        }
        self.assertEqual(
            {t.name for t in verify_types},
            {
                "EAGLE_VERIFY",
                "FROZEN_KV_MTP_VERIFY",
                "DFLASH_VERIFY",
                "NGRAM_VERIFY",
            },
            "a new verify layout was added; decide whether the DCP split "
            "serves it, and gate it at boot if not",
        )
        served = tb._DCP_VERIFY_SPEC_INPUT_TYPES
        gated_at_boot = {
            SpecInputType.NGRAM_VERIFY,
            SpecInputType.FROZEN_KV_MTP_VERIFY,
        }
        self.assertEqual(verify_types, served | gated_at_boot)


if __name__ == "__main__":
    unittest.main()
