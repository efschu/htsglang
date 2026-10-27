"""``Scheduler`` methods moved out of ``scheduler.py`` (refactor, behaviour-neutral): bodies are byte-identical,
the class stays reachable as ``Scheduler.<name>`` through the mixin. Logger name unchanged."""

from contextlib import contextmanager
from flliper.srt.managers.schedule_batch import ScheduleBatch
from flliper.srt.mem_cache import kv_cache_builder
import logging

logger = logging.getLogger("flliper.srt.managers.scheduler")


class SchedulerDraftKvProducerMixin:
    """Methods moved out of ``Scheduler`` (see module docstring)."""

    def _maybe_init_draft_kv_producer(self):
        """#1233 (C6): the draft-KV producer of Weg 2's prefill group.

        The single-rank pp group is built on EVERY rank (collective); the
        producer itself only on the stage `get_pp_group().is_last_rank`
        names -- derived from --pp-attn-stage-ratio, never "PP2".
        """
        if not getattr(self.server_args, "speculative_draft_kv_only", False):
            return
        from flliper.srt.distributed.parallel_state import (
            get_pp_group,
            initialize_draft_pp_group,
        )
        from flliper.srt.speculative.draft_kv_producer import DraftKvProducer

        initialize_draft_pp_group()
        if not get_pp_group().is_last_rank:
            return
        if self.draft_kv_producer_algorithm.is_dflash():
            # DFlash draft-KV producer: chunk ring + hash-keyed direct
            # publish into the draft arena (dflash_draft_kv_producer).
            from flliper.srt.speculative.dflash_draft_kv_producer import (
                DFlashDraftKvProducer,
            )

            self.draft_kv_producer = DFlashDraftKvProducer(
                self, self.draft_kv_producer_algorithm
            )
        else:
            self.draft_kv_producer = DraftKvProducer(self, self.draft_kv_producer_algorithm)
        embed_mib = self.draft_kv_producer.load_resident_embedding(self.server_args.model_path)
        draft_model = self.draft_kv_producer.draft_runner.model
        # #161: WELCHE Bytes liegen hier wirklich ZWEIMAL auf der Karte?
        # Der Zensus im ModelRunner (model_runner.py:2938) kann das nicht
        # sagen -- er laeuft ueber named_parameters() und sieht einen per
        # `lm_head_from_target()` GETEILTEN Tensor genauso wie einen
        # eigenen. Aus seiner Zeile "pp0tp0-draft ... embed_tokens 1.18,
        # lm_head 1.18" habe ich am 22.09. zweimal eine Zuordnung
        # abgeleitet und zweimal zurueckgenommen; beide Male war es eine
        # Vermutung, keine Messung.
        #
        # HIER, und nur hier, stehen BEIDE Modelle: das Ziel ueber den
        # Scheduler, der Draft ueber den Producer. Der ModelRunner hat
        # KEINE Ziel-Referenz (is_draft_worker ist ein blosses Flag), ein
        # peer-Argument dort waere immer None gewesen -- gebaut, nie
        # erreicht, die Klasse dieses ganzen Tages.
        try:
            from flliper.srt.model_executor.vram_family_census import (
                log_vram_family_census,
            )

            log_vram_family_census(
                draft_model,
                f"pp{self.ps.pp_rank}-draft-vs-target",
                "after draft load",
                peer=self.tp_worker.model_runner.model,
            )
        except Exception as _exc:  # noqa: BLE001 -- ein Zensus toetet nie
            logger.debug("[vram-census] #161 peer census skipped: %s", _exc)
        mtp_mib = sum(
            p.numel() * p.element_size() for n, p in draft_model.named_parameters()
            if "embed_tokens" not in n and "lm_head" not in n
        ) / float(2**20)
        cfg = self.draft_kv_producer.draft_runner.model_config
        logger.info(
            "PDFLIP DRAFT-KV-PRODUCER armed stage=%d/%d drafter=%s layout=v%d heads=%d "
            "head_dim=%d page_bytes=%d embed=resident mtp_mib=%.1f embed_mib=%.1f "
            "resident_mib=%.1f head_released_mib=%.1f head_deferred=%s "
            "nvml_delta_mib=%.1f tag_pool_inactive_mib=%.1f "
            "outside_torch_mib=%.1f default_pool_inactive_mib=%.1f "
            "card_free_mib=%.1f other_live_mib=%.1f "
            "embed_dtype=%s build_s=%.1f",
            self.draft_kv_producer.stage,
            self.draft_kv_producer.stages,
            kv_cache_builder.drafter_identity_hash(self.server_args),
            1,
            int(cfg.get_total_num_kv_heads()),
            int(cfg.head_dim),
            # per PAGE, not per token: under page_size > 1 a canonical draft
            # page carries page_size rows, and an instrument that says
            # otherwise misreads the store by that factor (#66).
            2
            * int(cfg.get_total_num_kv_heads())
            * int(cfg.head_dim)
            * max(1, int(getattr(self.server_args, "page_size", 1) or 1)),
            mtp_mib,
            embed_mib,
            # THREE instruments, each named on the line, none averaged with
            # another (fix 3, boot weg2dk3):
            #   resident_mib      the producer's own live weight bytes on this
            #                     stage, the shared target lm_head excluded.
            #                     W11 grades THIS one; -1 = unmeasured.
            #   head_released_mib the never-loaded lm_head table this handle
            #                     deleted (0 under tie_word_embeddings, where
            #                     no second table is ever built, AND 0 since
            #                     #1259 b, where it is never built either).
            #   head_deferred     #1259 (b): the table was never ALLOCATED, so
            #                     mem_get_info never charged it to this rank's
            #                     KV budget. Distinguishes that from the tie
            #                     case, which also reports head_released_mib=0.
            #   nvml_delta_mib    NVML free before the build minus after. Under
            #                     --enable-memory-saver this reads the BUILD,
            #                     not the residue: the weights live in the
            #                     saver's torch.cuda.MemPool, which
            #                     empty_cache() does not hand to the driver
            #                     (memsaver.md N4), and the KV pools reuse the
            #                     freed block out of that same pool.
            self.draft_kv_producer.resident_mib,
            self.draft_kv_producer.head_released_mib,
            self.draft_kv_producer.head_deferred,
            self.draft_kv_producer.nvml_delta_mib,
            # #66: what the private tag pools still cache -- empty_cache cannot
            # reach them, so NVML counts them and W11b must see them by name.
            getattr(self.draft_kv_producer, "tag_pool_inactive_mib", -1.0),
            # #66: die zwei Terme, die W11b fehlten, als die Rechnung 533,7 MiB
            # nicht erklaeren konnte (fnFL2v86). Sie beantworten die Frage, die
            # das Gate stellt, aber bisher nicht stellen KONNTE: liegt der Rest
            # INNERHALB von torch (ein Cache, den wir benennen koennen) oder
            # AUSSERHALB (CUDA-Kontext, cuBLAS-Workspaces, JIT-Kernel)?
            getattr(self.draft_kv_producer, "outside_torch_mib", -1.0),
            getattr(self.draft_kv_producer, "default_pool_inactive_mib", -1.0),
            # #66: NICHT Teil der Bilanz -- der Massstab, an dem W11b misst,
            # ob ein unerklaerter Rest die Karte bedroht oder nur den KV-Pool
            # schmaelert (den das Sizing danach ohnehin aus mem_get_info
            # ableitet, also mit diesen Bytes bereits rechnet).
            getattr(self.draft_kv_producer, "card_free_mib", -1.0),
            # #66: der fehlende Posten der W11b-Bilanz -- lebende Bytes, die
            # nicht dem Modell gehoeren (Attention-Workspace des Draft-Runners
            # voran). Weder Cache noch Gewicht, und deshalb von keinem der
            # vier bisherigen Terme sichtbar.
            getattr(self.draft_kv_producer, "other_live_mib", -1.0),
            self.draft_kv_producer.embed_dtype,
            self.draft_kv_producer.build_s,
        )
        # DFLASH-PRODUCE-ON-P (27B 2026-09-24, UNIFY S6 p_draft=cold): the
        # armed line above stays byte-identical (W11/W11b read it); the form is
        # named on its own line. The Scheduler has only ``self.ps.pp_rank``
        # (Draft-Zensus self.ps pin, UNIFY S2).
        if self.draft_kv_producer_algorithm.is_dflash():
            from flliper.srt.speculative.dflash_draft_kv_producer import (
                dflash_produce_off_line,
                dflash_produce_on_p,
            )

            if not dflash_produce_on_p():
                logger.info(
                    dflash_produce_off_line(
                        where=f"scheduler pp_rank={self.ps.pp_rank} (producer built, never asked)"
                    )
                )

    def _solo_draft_kv_pool_bytes(self) -> int:
        """Measured bytes of solo-resident draft KV pools on THIS rank.

        Walks the draft worker (including the cross-algorithm meta-worker's
        sub-workers) for draft model runners flagged as the solo host and
        sums their private KV pools. 0 on shadow ranks, on split placement,
        and without speculative decoding. Best-effort: sizing the balance
        post must never break boot."""
        total = 0
        try:
            roots = []
            dw = self.draft_worker
            if dw is not None:
                for attr in ("_primary", "_secondary"):
                    sub = getattr(dw, attr, None)
                    if sub is not None:
                        roots.append(sub)
                if not roots:
                    roots.append(dw)
            seen = set()
            for w in roots:
                dmr = getattr(w, "draft_model_runner", None)
                if (
                    dmr is None
                    or id(dmr) in seen
                    or not getattr(dmr, "is_draft_solo_host", False)
                ):
                    continue
                seen.add(id(dmr))
                pool = getattr(dmr, "token_to_kv_pool", None)
                if pool is None:
                    continue
                try:
                    v = pool.get_kv_size_bytes()
                    total += int(sum(v)) if isinstance(v, (tuple, list)) else int(v)
                except Exception:
                    continue
        except Exception:
            return 0
        return total

    def _draft_kv_producer_wants(self, batch: ScheduleBatch) -> bool:
        """#1233 (C7): the last stage's extend chunk, and nothing else.

        Fix 6 DELETED a trailing ``and not batch.forward_mode.is_idle()``:
        ``ForwardMode.IDLE.is_extend()`` is already False
        (forward_batch_info.py), so the clause could never change an answer.
        An unfalsifiable term reads as protection that is not there -- the
        wants-matrix row for IDLE passes with or without it, so the guard was
        pinning nothing while looking like it did.

        DFLASH-PRODUCE-ON-P (27B 2026-09-24, "P ohne draft rechnen"; UNIFY S6
        p_draft=cold): under ``FLLIPER_PDFLIP_DFLASH_PRODUCE=0`` a DFlash producer
        is built (cold-resident weights) but never asked -- no FULL capture,
        no ``produce()``, no ``publish_draft_rows_direct``. The NEXTN/MTP
        producer is unaffected.
        """
        if self.draft_kv_producer is None or not batch.forward_mode.is_extend():
            return False
        _is_dflash = getattr(
            getattr(self, "draft_kv_producer_algorithm", None), "is_dflash", None
        )
        if callable(_is_dflash) and _is_dflash():
            from flliper.srt.speculative.dflash_draft_kv_producer import (
                dflash_produce_on_p,
            )

            return dflash_produce_on_p()
        return True

    @contextmanager
    def _draft_kv_full_capture(self, batch: ScheduleBatch):
        """C7: the target chunk must hand the producer FULL hidden states.

        The restore is in a ``finally``: a producer that raises is a group
        STOP, but it must not first leave the batch in FULL capture for
        whatever the scheduler touches on the way down.
        """
        from flliper.srt.model_executor.forward_batch_info import CaptureHiddenMode

        saved = batch.capture_hidden_mode
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        try:
            yield
        finally:
            batch.capture_hidden_mode = saved

    def _draft_kv_produce(self, batch: ScheduleBatch, batch_result) -> None:
        """Run the draft-extend of this chunk on the producer and emit L1."""
        hidden = getattr(batch_result.logits_output, "hidden_states", None)
        next_ids = batch_result.next_token_ids
        if hidden is None or next_ids is None:
            raise RuntimeError(
                "PDFLIP DRAFT-KV-PRODUCE: the target chunk returned no FULL hidden "
                "states or no sampled ids; the producer cannot write draft rows "
                "for this chunk (never silently skipped)."
            )
        saved_algo, saved_spec = batch.spec_algorithm, batch.spec_info
        batch.spec_algorithm = self.draft_kv_producer_algorithm
        try:
            terms = self.draft_kv_producer.produce(batch, hidden, next_ids)
        finally:
            batch.spec_algorithm = saved_algo
            batch.spec_info = saved_spec
        rid = batch.reqs[0].rid if batch.reqs else "?"
        chunked = batch.chunked_req is not None
        n = self.draft_kv_producer._chunks
        if n <= 64 or n % 64 == 0 or not chunked:
            logger.info(
                "PDFLIP DRAFT-KV-PRODUCE rid=%s chunk=%d/%s tokens=%d rows=[%d,%d) tail=%s "
                "ms=%.1f peak_mib=%.1f (n=%d)",
                rid,
                n,
                "?",
                terms["rows"],
                self.draft_kv_producer._rows - terms["rows"],
                self.draft_kv_producer._rows,
                "next_prompt" if chunked else "sampled",
                terms["ms"],
                terms["peak_mib"],
                n,
            )
