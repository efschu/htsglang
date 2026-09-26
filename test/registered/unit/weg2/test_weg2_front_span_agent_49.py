"""#49 -- the FRONT-SPAN gap: agent turns ran over P + 2 flips although D held the prefix.

MEASURED, boot dkr27bbar1agent09251922 (27B INT8, cu130 container, front log
``boot_weg2_dkr27bbar1agent09251922_e914e89fde_0925_192304.front.log``,
19:32-19:45Z): one Claude-Code agent (Anthropic /v1/messages, tools,
thinking, streaming) sent 45 generations + 48 count_tokens. 49 generations
went over P, 64 flips in 13 min, although every D leg 2 read the prefix back
(e.g. weg2-12-8 prompt_tokens=23830 cached_tokens=23828) and the real new
tail per turn was 0.2k-6k tokens against X=4096.

THREE DEFECTS, each visible in the log, each pinned below:

(A) ``request_text`` rendered ``tools`` LAST. The Qwen3.8 template renders the
    tool schemas FIRST (inside the system turn, chat_template.jinja:57-67), so
    D's radix prefix covers them, but the front's character prefix match
    stopped at the end of the previous turn's messages and priced the whole
    tool block as uncached on EVERY turn. Measured: the reference entry's
    unmatched tail is 13,302 chars on all 30 main turns (= the tool block,
    ~4.4k est tokens) -> est_uncached 5.4k-11k > X=4096 -> LONG.
(B) A turn D served DIRECTLY recorded only its ``cached_tokens`` (what D held
    BEFORE serving) and taught the front nothing about the full prompt D now
    holds in its radix. The credit therefore lagged a turn behind.
(C) ``price_remainder`` subtracted a MEASURED token credit from a chars/3
    ESTIMATE of the whole prompt. At 3.1 real chars/token (measured) the
    prefix alone was over-priced by ~2k tokens at 70k context, enough to push a
    2.5k real tail over X=4096.

THE LAW IS UNTOUCHED (memory kein-d-direct-prefill-ueber-x): real uncached
work above X still routes LONG -- pinned by
``test_a_real_tail_above_x_still_routes_long`` and by the replay's two turns
whose real tail exceeds X (weg2-10-7 4824, weg2-30-20 6117).

DANGER DIRECTION = over-crediting. The held credit (B) is valid ONLY in the
epoch D served the text in and only while D is awake and serving: D flushes
its radix when it sleeps, and the epoch changes on every flip. Mutants:
  M1 held credit across an epoch      -> test_held_credit_is_bound_to_the_epoch
  M2 held credit while D is not awake -> test_held_credit_needs_an_epoch
  M3 retraction pop removed           -> test_zero_without_serve_still_retracts
  M5 a 2k over-credit on the prefix   -> the replay + test_same_tools_next_turn...
  M4 changed tools still credited     -> test_changed_tools_block_prices_whole
"""

import inspect
import json

import pytest

from sglang.srt.weg2 import front as F
from sglang.srt.weg2.front import SpanLRU, price_remainder, request_text, serviceable_route



@pytest.fixture(autouse=True)
def _front_span_49_on(monkeypatch):
    """Unified tree (FS 26.09.): #49 sits behind SGLANG_WEG2_FRONT_SPAN_49,
    default off. Every test in this file pins the switched-ON behaviour; the
    OFF path is pinned in test_weg2_front_span_inflight_fs.py."""
    monkeypatch.setenv("SGLANG_WEG2_FRONT_SPAN_49", "1")


X = 4096  # the boot's front X (--tp-prefill-max-tokens 4096, never re-solved)
D_GATE = 12288  # D's own W50 riegel (--x-ceiling-tokens 12288)
CARRIER_MAX = 648806  # from the boot's LONG lines
TOOL_BLOCK_CHARS = 13302  # measured: the reference entry's unmatched tail, every main turn

# (rid, kind, chars of request_text, prompt_tokens D reported), arrival order,
# verbatim from the front log's ROUTE-VERDICT (chars=) and WEG2-SERVED D lines.
# "side" = Claude Code's parallel side request: the preceding main turn's
# context + one ~585-char user message, never continued by the next turn.
MEASURED = [
    ("weg2-9-6", "main", 58033, 18112),
    ("weg2-10-7", "main", 72036, 22936),
    ("weg2-12-8", "main", 74614, 23830),
    ("weg2-14-9", "main", 76908, 24698),
    ("weg2-14-10", "side", 77417, 24836),
    ("weg2-16-11", "main", 87904, 28391),
    ("weg2-18-12", "main", 96543, 31314),
    ("weg2-20-13", "main", 104619, 33949),
    ("weg2-21-14", "side", 105205, 34104),
    ("weg2-22-15", "main", 105072, 34123),
    ("weg2-24-16", "main", 113779, 37093),
    ("weg2-26-17", "main", 121112, 39620),
    ("weg2-27-18", "side", 121710, 39778),
    ("weg2-28-19", "main", 131259, 43084),
    ("weg2-30-20", "main", 149283, 49201),
    ("weg2-32-21", "main", 152081, 50066),
    ("weg2-32-22", "side", 152664, 50222),
    ("weg2-34-23", "main", 161202, 53032),
    ("weg2-36-24", "main", 164504, 54171),
    ("weg2-37-25", "side", 165098, 54328),
    ("weg2-38-26", "main", 166641, 54936),
    ("weg2-40-27", "main", 174400, 57433),
    ("weg2-41-28", "side", 174994, 57590),
    ("weg2-42-29", "main", 176860, 58235),
    ("weg2-44-30", "main", 178616, 58888),
    ("weg2-45-31", "side", 179199, 59046),
    ("weg2-46-32", "main", 187142, 61147),
    ("weg2-47-33", "side", 187725, 61302),
    ("weg2-48-34", "main", 194189, 63572),
    ("weg2-50-35", "side", 194776, 63728),
    ("weg2-50-36", "main", 205261, 66480),
    ("weg2-52-37", "main", 206229, 66801),
    ("weg2-54-38", "main", 208100, 67553),
    ("weg2-55-39", "side", 208684, 67708),
    ("weg2-56-40", "main", 210400, 68375),
    ("weg2-58-41", "side", 210990, 68531),
    ("weg2-59-42", "main", 215591, 70064),
    ("weg2-60-43", "main", 223710, 72790),
    ("weg2-61-44", "side", 224291, 72944),
    ("weg2-62-47", "main", 231069, 75272),
    ("weg2-63-49", "side", 231642, 75425),
    ("weg2-64-50", "main", 231982, 75615),
    ("weg2-66-52", "main", 232630, 75873),
    ("weg2-67-53", "side", 233206, 76028),
    ("weg2-68-54", "main", 233481, 76155),
]


# --------------------------------------------------------------------------
# the replay: Claude-Code-shaped Anthropic bodies of the measured lengths
# --------------------------------------------------------------------------


def _tools(target_chars: int):
    """A tool list whose request_text block is exactly ``target_chars`` long."""
    def block(tools):
        return len("tools:" + json.dumps(tools, ensure_ascii=False, sort_keys=True) + "\n")

    tools = [
        {"name": f"Tool{i}", "description": "d",
         "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}}
        for i in range(20)
    ]
    pad = target_chars - block(tools)
    assert pad > 0
    tools[0]["description"] = "d" + "x" * pad
    assert block(tools) == target_chars
    return tools


def _msg_line(role: str, content: str) -> int:
    return len(f"{role}:{content}\n")


def _filler(tag: str, n: int) -> str:
    body = (tag + " ") * (n // (len(tag) + 1) + 1)
    return body[:n]


def build_bodies(tool_block_chars: int = TOOL_BLOCK_CHARS):
    """One Anthropic body per measured request, lengths matched to the chars."""
    tools = _tools(tool_block_chars)
    system = "You are Claude Code. " * 200
    base = {"model": "Qwen3.8-27B", "system": system, "tools": tools, "stream": True}
    msgs = []
    out = []
    for i, (rid, kind, chars, pt) in enumerate(MEASURED):
        cur = len(request_text({**base, "messages": msgs}))
        if kind == "main":
            need = chars - cur
            if not msgs:
                content = _filler(f"u{i}", need - _msg_line("user", ""))
                msgs = [{"role": "user", "content": content}]
            else:
                # assistant answer + tool_result, as a Claude-Code turn grows
                a = max(1, need // 2)
                a_c = _filler(f"a{i}", a - _msg_line("assistant", ""))
                u_c = _filler(f"r{i}", need - a - _msg_line("user", ""))
                msgs = msgs + [{"role": "assistant", "content": a_c},
                               {"role": "user", "content": u_c}]
            body = {**base, "messages": msgs}
        else:
            need = chars - cur
            extra = _filler(f"s{i}", need - _msg_line("user", ""))
            body = {**base, "messages": msgs + [{"role": "user", "content": extra}]}
        assert len(request_text(body)) == chars, (rid, len(request_text(body)), chars)
        out.append((rid, kind, body, pt))
    return out


def _price(text, spans, epoch):
    try:
        return price_remainder(text, spans, epoch=epoch)
    except TypeError:  # the pre-#49 signature (red-first run on 3babf514fc)
        return price_remainder(text, spans)


def _record(spans, text, ct, pt, epoch):
    try:
        spans.record_presence(text, ct, prompt_tokens=pt, held_epoch=epoch)
    except TypeError:  # the pre-#49 signature: ct only
        spans.record_presence(text, ct)


def replay(bodies):
    """Drive the front's pricing + leg-2 recording over the measured sequence.

    The group model is the boot's: idle layout P; a LONG while D is awake
    costs D->P + P->D (2 flips, radix flushed, P's KV carried back so D's
    leg 2 reads pt-2); a SHORT on the awake D is served directly and D reads
    back the longest text it served in this epoch that prefixes the request.
    Think times in the log are <= 2.3 s after every answer (D-hold 10 s), so
    D stays awake between turns. Returns (rows, flips).
    """
    spans = SpanLRU()
    awake, epoch, flips = "P", 0, 0
    radix = []  # (text, pt) D holds in this epoch
    rows = []
    for rid, kind, body, pt in bodies:
        text = request_text(body)
        rem, est, known = _price(text, spans, epoch if awake == "D" else None)
        carrier_est = int(len(text) / F.CARRIER_CHARS_PER_TOKEN) + 1
        route = serviceable_route(rem, carrier_est, X, CARRIER_MAX)
        # what D would have to prefill if it served this now: the ground truth
        # the route is judged against (D's radix = texts served this epoch)
        d_rest = pt - max([p for t, p in radix if text.startswith(t)] or [0]) \
            if awake == "D" else pt
        if awake == "D" and route == "short":
            ct = pt - d_rest
            path = "D-direct"
            radix.append((text, pt))
        else:
            if awake == "D":
                flips += 2
                epoch += 2
            else:
                flips += 1
                epoch += 1
            awake = "D"
            ct = pt - 2
            path = "P+flips"
            radix = [(text, pt)]
        _record(spans, text, ct, pt, epoch)
        rows.append(dict(rid=rid, kind=kind, est_uncached=rem, route=route,
                         path=path, d_rest=d_rest, pt=pt, ct=ct))
    return rows, flips


# --------------------------------------------------------------------------
# (1) the replay of the measured sequence
# --------------------------------------------------------------------------


def test_replay_agent_turns_stay_on_d():
    """THE acceptance: no flip for a turn whose real uncached rest fits X.

    Before (3babf514fc): 44 of 44 requests after the first price LONG
    (est_uncached 5.4k-11k -- the replay reproduces the log's own numbers,
    e.g. 6173 / 6057 for weg2-12-8 / weg2-14-9), 89 flips in this model.
    After: 42 D-direct, the 2 turns with a real rest > X over P, 5 flips.
    """
    rows, flips = replay(build_bodies())
    first = rows[0]
    assert first["path"] == "P+flips"  # the cold 18k first turn: P's job
    warm = rows[1:]
    fits = [r for r in warm if r["d_rest"] <= 0.9 * X]
    over = [r for r in warm if r["d_rest"] > X]
    moved = [r["rid"] for r in fits if r["path"] != "D-direct"]
    assert not moved, f"turns whose real rest fits X still flipped: {moved}"
    # the law: real work above X still goes over P
    assert [r["rid"] for r in over] == ["weg2-10-7", "weg2-30-20"]
    assert all(r["path"] == "P+flips" for r in over)
    # and no D-direct turn prefills beyond D's own riegel
    assert all(r["d_rest"] <= D_GATE for r in warm if r["path"] == "D-direct")
    # the estimate tracks D's real rest (measured tails, 3.1 chars/token)
    assert all(abs(r["est_uncached"] - r["d_rest"]) <= 800
               for r in warm if r["path"] == "D-direct")
    # 1 (P->D for the cold first turn) + 2 x 2 (the two real LONG turns)
    assert flips == 5, flips


def test_replay_before_after_numbers_are_printed(capsys):
    """Not an assertion on the tree -- the before/after table for the record."""
    rows, flips = replay(build_bodies())
    for r in rows:
        print(f"{r['rid']:<11} {r['kind']:<4} est_uncached={r['est_uncached']:>6} "
              f"d_rest={r['d_rest']:>6} route={r['route']:<5} {r['path']}")
    print(f"flips={flips}")
    assert rows


# --------------------------------------------------------------------------
# (2) the pieces
# --------------------------------------------------------------------------


def test_tools_render_before_system_and_messages():
    """(A): the order of the model's own template, both wire shapes."""
    tools = [{"name": "Bash", "description": "run", "input_schema": {}}]
    anth = {"system": "S", "messages": [{"role": "user", "content": "hi"}], "tools": tools}
    oai = {"messages": [{"role": "system", "content": "S"},
                        {"role": "user", "content": "hi"}], "tools": tools}
    t = request_text(anth)
    assert t.startswith("tools:"), t[:40]
    assert t.index("tools:") < t.index("system:S") < t.index("user:hi")
    assert request_text(oai) == t


def test_same_tools_next_turn_prices_only_the_new_tail():
    """(A)+(B): a turn that only appends is priced by its appended chars."""
    bodies = build_bodies()
    (_, _, b0, pt0), (_, _, b1, pt1) = bodies[1], bodies[2]  # 10-7 -> 12-8
    spans = SpanLRU()
    t0, t1 = request_text(b0), request_text(b1)
    spans.record_presence(t0, pt0 - 2, prompt_tokens=pt0, held_epoch=5)
    rem, est, known = price_remainder(t1, spans, epoch=5)
    assert known
    tail_est = (len(t1) - len(t0)) / F.CHARS_PER_TOKEN
    assert rem <= tail_est + 3, (rem, tail_est)
    assert rem < X
    # the measured instrument on this very pair said 6173 (> X)


def test_changed_tools_block_prices_whole():
    """M4: a different tool list is a different prefix -- no credit past it."""
    spans = SpanLRU()
    a = {"system": "S" * 3000, "messages": [{"role": "user", "content": "q" * 30000}],
         "tools": [{"name": "A", "description": "x" * 9000}]}
    b = {**a, "tools": [{"name": "B", "description": "x" * 9000}]}
    ta, tb = request_text(a), request_text(b)
    spans.record_presence(ta, 14000, prompt_tokens=14000, held_epoch=3)
    rem, est, _ = price_remainder(tb, spans, epoch=3)
    assert rem > X, "the credit must stop where the tool schemas differ"


def test_held_credit_is_bound_to_the_epoch():
    """M1: D flushes its radix on sleep; a flip ends the held credit."""
    spans = SpanLRU()
    t = "p" * 60000
    spans.record_presence(t, 0, prompt_tokens=20000, held_epoch=7)
    same, est, known = price_remainder(t + "n" * 300, spans, epoch=7)
    assert known and same <= 101
    other, est2, known2 = price_remainder(t + "n" * 300, spans, epoch=9)
    assert not known2 and other == est2, "a held credit must not survive a flip"


def test_held_credit_needs_an_epoch():
    """M2: the route passes no epoch unless D is awake and serving."""
    spans = SpanLRU()
    t = "p" * 60000
    spans.record_presence(t, 0, prompt_tokens=20000, held_epoch=7)
    rem, est, known = price_remainder(t, spans)
    assert not known and rem == est
    src = inspect.getsource(F.Front.handle_generate)
    assert 'self.awake == "D"' in src and 'self.state == "serving"' in src
    assert "price_remainder(text, self.spans, epoch=" in src


def test_zero_without_serve_still_retracts():
    """M3: the #1324 rule stands for a reading without a D serve."""
    spans = SpanLRU()
    t = "w" * 30000
    spans.record_presence(t, 9000, prompt_tokens=9500)
    assert price_remainder(t, spans)[2] is True
    spans.record_presence(t, 0, prompt_tokens=9500)
    rem, est, known = price_remainder(t, spans)
    assert not known and rem == est


def test_a_real_tail_above_x_still_routes_long():
    """kein-d-direct-prefill-ueber-x: a warm D does not prefill a 20k paste."""
    spans = SpanLRU()
    t = "c" * 150000
    spans.record_presence(t, 50000, prompt_tokens=50000, held_epoch=4)
    rem, est, known = price_remainder(t + "z" * 60000, spans, epoch=4)
    assert known and rem >= 20000
    assert serviceable_route(rem, est, X, CARRIER_MAX) == "long"


def test_prefix_is_priced_in_measured_tokens_not_chars():
    """(C): the prefix D holds is priced by its own prompt_tokens.

    3.1 chars/token (measured on this agent): chars/3 over-prices a 70k-token
    prefix by ~2.3k -- the whole margin of X.
    """
    spans = SpanLRU()
    t = "k" * 217000  # 70000 tokens at 3.1
    spans.record_presence(t, 70000, prompt_tokens=70000)
    rem, est, _ = price_remainder(t + "m" * 7500, spans)
    assert rem <= 2501, rem  # the tail alone, at chars/3
    # a prefix D holds only PART of stays priced by what it lacks
    spans.record_presence(t, 60000, prompt_tokens=70000)
    rem2, _, _ = price_remainder(t + "m" * 7500, spans)
    assert rem2 >= 10000 + 2500 - 1


def test_leg2_records_prompt_tokens_and_the_serving_epoch():
    """The wiring: both D leg-2 branches record pt, and hold it only on a serve."""
    src = inspect.getsource(F.Front.leg2)
    assert src.count("spans.record_presence(text, ct, prompt_tokens=(pt if _fs49 else 0),") == 2
    assert src.count("held_epoch=") >= 2
    assert "spans.record_presence(text, pt" not in src
