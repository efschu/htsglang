"""``Scheduler`` methods moved out of ``scheduler.py`` (refactor, behaviour-neutral): bodies are byte-identical,
the class stays reachable as ``Scheduler.<name>`` through the mixin. Logger name unchanged."""

from flliper.srt.managers.pdflip_pass_timer import read_ms as _pt_read
import logging

logger = logging.getLogger("flliper.srt.managers.scheduler")


class SchedulerPostWakeMixin:
    """Methods moved out of ``Scheduler`` (see module docstring)."""

    def _pdflip_post_wake_pass_log(self, batch) -> None:
        """Wake-Parallel item 2 (user 18.09.): the first 8 passes after DORMANT
        cleared, phase by phase -- schedule (get_next_batch_to_run incl.
        prepare_for_extend), run (run_batch launch), prefetch, process_input,
        and the wall gap since the previous pass -- so the ~2 s between the
        hold release and the first decode (xsn311 flip 8) get a name.
        Armed by the wake (`_pdflip_post_wake_pass_n = 0`), disarms itself."""
        n = getattr(self, "_pdflip_post_wake_pass_n", None)
        if n is None:
            return
        if n >= 8:
            self._pdflip_post_wake_pass_n = None
            return
        if n == 0:
            self._pdflip_arm_wake_round_census()
        self._pdflip_post_wake_pass_n = n + 1
        import time as _t
        now = _t.perf_counter()
        prev = getattr(self, "_pdflip_post_wake_t", None)
        self._pdflip_post_wake_t = now
        try:
            mode = getattr(getattr(batch, "forward_mode", None), "name", "?")
            bs = int(getattr(batch, "batch_size", lambda: 0)()) if callable(getattr(batch, "batch_size", None)) else len(getattr(batch, "reqs", ()) or ())
        except Exception:  # noqa: BLE001
            mode, bs = "?", -1
        _g = getattr(self, "_pdflip_gnbp_ms", None) or (-1.0, -1.0, -1.0)
        self._pdflip_gnbp_ms = None
        # xsn325: the host cost per admission (add_one_req wall, host hit) and
        # the init_load_back share of it (the DMA issue), so init_new splits
        # into DMA and host work per request.
        _arl = getattr(self, "_pdflip_gnbp_addreq", None) or []
        self._pdflip_gnbp_addreq = None
        _addreq = " ".join(f"{rid}:{ms:.0f}ms/hit{hit}" for rid, ms, hit in _arl[:8]) or "-"
        _irl = getattr(self, "_pdflip_gnbp_initr", None) or []
        self._pdflip_gnbp_initr = None
        _initr = " ".join(f"{rid}:{ms:.0f}ms" for rid, ms in _irl[:8]) or "-"
        try:
            from flliper.srt.managers import schedule_policy as _sp
            _lb_ms, _lb_n = float(_sp.PDFLIP_ADMIT_T.get("lb_ms", 0.0)), int(_sp.PDFLIP_ADMIT_T.get("lb_n", 0))
            _sp.PDFLIP_ADMIT_T["lb_ms"] = 0.0
            _sp.PDFLIP_ADMIT_T["lb_n"] = 0
        except Exception:  # noqa: BLE001
            _lb_ms, _lb_n = -1.0, -1
        # read_ms resets what it reads: read each pass term ONCE, for the line
        # and for rankstats.last_post_wake alike.
        _sched_ms, _run_ms = _pt_read(self, "_1466_schedule_ms"), _pt_read(self, "_1466_run_ms")
        _pf_ms, _pi_ms = _pt_read(self, "_1474_prefetch_ms"), _pt_read(self, "_1475_process_input_ms")
        _gap_ms = ((now - prev) * 1000.0) if prev is not None else -1.0
        logger.info(
            "PDFLIP-POST-WAKE-PASS n=%d mode=%s bs=%d gap_ms=%.0f schedule_ms=%.0f run_ms=%.0f "
            "prefetch_ms=%.0f proc_input_ms=%.0f admission_ms=%.0f init_new_ms=%.0f prepare_ms=%.0f ready_ms=%.0f "
            "addreq=[%s] init_load_back_ms=%.0f/%d init_next_round=[%s] waiting=%d admit=%s admit_partial=%s "
            "(gap = wall since the previous pass; schedule = get_next_batch_to_run incl. the three "
            "prefill terms; run is the launch, the forward itself overlaps; admit = the admission's "
            "own gate / skip census of this pass, BA 28.09.)",
            n, mode, bs, _gap_ms, _sched_ms, _run_ms, _pf_ms, _pi_ms,
            _g[0], _g[1], _g[2], float(getattr(self, '_pdflip_ready_ms', -1.0) or -1.0), _addreq, _lb_ms, _lb_n, _initr,
            len(getattr(self, "waiting_queue", None) or ()),
            str(getattr(self, "_admission_decline_note", None) or "-")[:240],
            str(getattr(self, "_admission_partial_note", None) or "-")[:240],
        )
        # DASHBOARD-AUS-IPC: the same census as rankstats.last_post_wake (one
        # assignment here; the rankstats timer writes it -- never a decode round).
        from flliper.srt.pdflip import rankstats as _pdflip_rankstats

        _pdflip_rankstats.note_post_wake({
            "n": n, "mode": mode, "bs": bs, "t": round(_t.time(), 3), "gap_ms": round(_gap_ms, 1),
            "schedule_ms": _sched_ms, "run_ms": _run_ms, "prefetch_ms": _pf_ms, "proc_input_ms": _pi_ms,
            "admission_ms": _g[0], "init_new_ms": _g[1], "prepare_ms": _g[2],
            "waiting": len(getattr(self, "waiting_queue", None) or ()),
        })
        self._admission_partial_note = None
