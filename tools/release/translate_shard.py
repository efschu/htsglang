#!/usr/bin/env python3
"""FL7 translation driver (RM 28.09.): translate english_audit units through one seat of the split router.

Worker contract as RENAME_PLAN 8.5: the translation is the COMPLETE replacement token (same quotes/prefix,
same '#'), every machine token kept, glossary for domain words. The gate (english_audit.py check) carries the
guarantee; this driver only refuses what it can already see is wrong, and retries it:
  * the reply is parsed leniently (the first JSON array in the text);
  * every unit is validated: same comment marker / string prefix and quotes, same multiset of machine tokens
    (placeholders, numbers, UPPER-DASH markers, #refs, `backticks`, --flags);
  * a batch whose reply is unusable is split in halves (down to one unit); a unit that still fails validation
    after RETRIES attempts stays untranslated (listed in the timing file) -- it keeps its German text.
  * translation memory: a unit whose exact text was translated before (--tm file) is reused, not re-sent.

usage: translate_shard.py <units.jsonl> <out.jsonl> <timing.jsonl> [model] [kinds] [tm.jsonl]
  kinds: comma list (e.g. log,help,doc) or "all"
"""
import collections, json, os, re, sys, time, urllib.request
KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
from english_audit import machine_tokens  # THE gate's token rule, not a look-alike
MUST_KEEP = [l.strip() for l in open(os.path.join(KIT, "data", "must_keep.txt")) if l.strip() and not l.startswith("#")]

SRC, OUT, TIM = sys.argv[1:4]
MODEL = sys.argv[4] if len(sys.argv) > 4 else "Qwen3.8-27B-cachy-think"
KINDS = None if len(sys.argv) <= 5 or sys.argv[5] == "all" else set(sys.argv[5].split(","))
TM_FILE = sys.argv[6] if len(sys.argv) > 6 else None
URL = "http://127.0.0.1:30099/v1/messages"
BATCH_UNITS, BATCH_CHARS, RETRIES = 40, 24000, 2
GLOSS = json.load(open(os.path.join(KIT, "data", "de_en_subwords.json")))
GLOSS = GLOSS.get("map", GLOSS)

RULES = """You translate German source-code text (comments, docstrings, log/raise messages, argparse help, markdown lines) into English.
Input: a JSON array of units {"id", "kind", "text"}. "text" is the EXACT source token: a whole comment (with its '#'),
or a whole string literal (with its quotes and prefix), or a docstring (with its triple quotes), or a markdown line.
Output: ONLY a JSON array of {"id", "translation"} with one entry per input id -- no prose, no code fence.
Rules:
1. "translation" is the COMPLETE replacement token: same comment marker and indentation of continuation lines,
   same quotes/prefix (f, r, b), same line structure for docstrings (translate line by line, keep line count if possible).
2. Keep EVERY machine token byte for byte: UPPER-DASH markers (e.g. WEG2-FLIP), key= fields, %s/%d/{...} placeholders,
   #ticket refs (#1234), numbers, `backticked` text, CONST_NAMES, snake_case/camelCase/dotted identifiers, --flags, paths.
3. Quoted user orders: translate and append " (user order, translated)".
3b. Words written in ALL CAPS inside log/help strings (NICHT, OHNE, KARTE, ...) are matched by evaluators: keep them exactly as they are.
4. If a unit is already English, return it unchanged.
5. Domain glossary (German -> English): @@GLOSS@@
"""

_MACHINE = re.compile(r"%[-+ #0-9.]*[sdifrxeEgG%]|\{[^{}\s]*\}|#\d+|`[^`]+`|--[a-z][a-z0-9-]*|\b\d+(?:[.,]\d+)*\b"
                      r"|\b[A-Z0-9]+(?:-[A-Z0-9]+)+\b")
_STRPFX = re.compile(r'^\s*([rRbBuUfF]{0,2})("""|\'\'\'|"|\')')


def machine(s):
    return collections.Counter(_MACHINE.findall(s))


def shape(kind, s):
    t = s.lstrip()
    if kind == "comment":
        return ("#", [ln[: len(ln) - len(ln.lstrip())] + "#" for ln in s.splitlines() if ln.strip()])
    if kind == "doc":
        return ("doc",)
    m = _STRPFX.match(s)
    return ("str", m.group(1).lower(), m.group(2), s.rstrip().endswith(m.group(2))) if m else ("other", t[:1])


def valid(u, tr):
    if not isinstance(tr, str) or not tr.strip():
        return "empty"
    if shape(u["kind"], u["text"]) != shape(u["kind"], tr):
        return "shape"
    if machine_tokens(u["text"]) != machine_tokens(tr):
        return "machine-tokens"
    if u["kind"] not in ("comment", "docstring", "doc") and any(u["text"].count(k) != tr.count(k) for k in MUST_KEEP):
        return "must-keep"   # evaluators match runtime output: the gate binds log/help/string units to the list
    return None


def call(units):
    gl = {k: v for k, v in GLOSS.items() if any(k in u["text"].lower() for u in units)}
    body = {"model": MODEL, "max_tokens": 16000,
            "system": RULES.replace("@@GLOSS@@", json.dumps(gl, ensure_ascii=False)),
            "messages": [{"role": "user", "content": json.dumps(
                [{"id": u["_id"], "kind": u["kind"], "text": u["text"]} for u in units], ensure_ascii=False)}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "content-type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": "x"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    txt = "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text")
    i, j = txt.find("["), txt.rfind("]")
    res = json.loads(txt[i:j + 1]) if 0 <= i < j else None
    if not isinstance(res, list):
        raise ValueError("no JSON array in reply")
    return {int(r["id"]): r.get("translation") for r in res if isinstance(r, dict) and "id" in r}, \
        time.time() - t0, d.get("usage", {})


units = [json.loads(l) for l in open(SRC)]
if KINDS:
    units = [u for u in units if u["kind"] in KINDS]
for i, u in enumerate(units):
    u["_id"] = i
tm = {}
if TM_FILE:
    try:
        for l in open(TM_FILE):
            r = json.loads(l)
            if r.get("translation"):
                tm[r["text"]] = r["translation"]
    except FileNotFoundError:
        pass
done, reasons = {}, collections.Counter()
todo = []
for u in units:
    if u["text"] in tm and not valid(u, tm[u["text"]]):
        done[u["_id"]] = tm[u["text"]]; reasons["tm-hit"] += 1
    else:
        todo.append(u)
batches, cur, cc = [], [], 0
for u in todo:
    if cur and (len(cur) >= BATCH_UNITS or cc + len(u["text"]) > BATCH_CHARS):
        batches.append(cur); cur, cc = [], 0
    cur.append(u); cc += len(u["text"])
if cur:
    batches.append(cur)

T0 = time.time()
queue = [(b, 0) for b in batches]
with open(TIM, "w") as tf:
    def log(rec):
        rec["wall_s"] = round(time.time() - T0, 1)
        tf.write(json.dumps(rec) + "\n"); tf.flush(); print(json.dumps(rec), flush=True)
    while queue:
        b, attempt = queue.pop(0)
        rec = {"units": len(b), "chars": sum(len(u["text"]) for u in b), "attempt": attempt}
        try:
            got, dt, usage = call(b)
            bad = []
            for u in b:
                why = valid(u, got.get(u["_id"]))
                if why:
                    bad.append(u); reasons[why] += 1
                else:
                    done[u["_id"]] = got[u["_id"]]
            rec.update(ok=True, s=round(dt, 1), out_tokens=usage.get("output_tokens"), bad=len(bad))
        except Exception as e:
            bad = list(b); reasons["reply"] += 1
            rec.update(ok=False, err=f"{type(e).__name__}: {str(e)[:160]}")
        log(rec)
        if bad and attempt < RETRIES:
            if len(bad) > 1:   # split, so one unit the model chokes on cannot sink its neighbours again
                h = (len(bad) + 1) // 2
                queue += [(bad[:h], attempt + 1), (bad[h:], attempt + 1)]
            else:
                queue.append((bad, attempt + 1))
        elif bad:
            log({"gave_up": [u["_id"] for u in bad]})
with open(OUT, "w") as of:
    for u in units:
        o = {k: v for k, v in u.items() if k != "_id"}
        o["translation"] = done.get(u["_id"])
        of.write(json.dumps(o, ensure_ascii=False) + "\n")
if TM_FILE:
    with open(TM_FILE, "a") as t:
        for u in units:
            if u["_id"] in done and u["text"] not in tm:
                t.write(json.dumps({"text": u["text"], "translation": done[u["_id"]]}, ensure_ascii=False) + "\n")
print(json.dumps({"TOTAL_S": round(time.time() - T0, 1), "units": len(units), "translated": len(done),
                  "untranslated": len(units) - len(done), "reasons": dict(reasons)}))
