"""Language switch (user 08.10.): English is the source text of the page, German a translation memory
(static/i18n_de.json, applied by static/i18n.js).  What this pins:

* every text the STATIC markup of index.html / weg2.html shows has a German entry -- a new English label without one
  stays English in the German view, and this is where it shows up;
* the texts this change brought (history, sessions, TTFT causes) are translated;
* the memory is well formed and the page loads the modules in the order they need.
"""

import html.parser
import json
import os
import re

from rigdash import server

STATIC = os.path.join(os.path.dirname(os.path.abspath(server.__file__)), "static")

#: shown as they are in both languages: names, file names, identifiers, units
SAME_IN_BOTH = {"fLLiper", "Grafana", "JSON", "MHz", "MHz · NVML CLOCK_MEM", "--edition release", "state.json", "IPC", "events.jsonl",
                "rankstate/<G>/*.rankstats", "prefill.compute_ms", "Decode tok/s", "decode.running", "decode.gpu_ms", "GET /weg2/state",
                "/metrics", "TTFT", "LEG2-FIRST-CONTENT", "state.json front.arrival_seat.ttft_*", "front.ttft_by_via", "weg2_front_ttft_*",
                "state.json: groups.{P,D}.launch, launch.front, launch.container", "English", "Deutsch", "host_acceptance.sh",
                "docker images", "<fenster-id>", "Transport", "Boots", "prefill.new_tokens"}


def memory():
    with open(os.path.join(STATIC, "i18n_de.json"), encoding="utf-8") as fh:
        return json.load(fh)


def german(core, mem):
    """The rule of i18n.js tr(): the whole text, else every phrase / word of the memory in it (longest first)."""
    exact = dict(mem["de"], **mem["words"])
    if core in exact:
        return exact[core]
    keys = sorted((k for k in exact if re.match(r"[A-Za-z]", k) and (k in mem["words"] or (len(k) >= 8 and (" " in k or len(k) >= 14)))), key=len, reverse=True)
    out, i = "", 0
    while i < len(core):
        for k in keys:
            if core.startswith(k, i) and not re.match(r"[A-Za-z0-9_]", core[i + len(k):i + len(k) + 1]) and not (i and re.match(r"[A-Za-z0-9_]", core[i - 1])):
                out += exact[k]
                i += len(k)
                break
        else:
            out += core[i]
            i += 1
    return out


class Texts(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.found = [], []

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)
        if tag not in ("script", "style", "meta", "link"):
            self.found += [v for k, v in attrs if k in ("title", "placeholder", "aria-label", "alt") and v]

    def handle_endtag(self, tag):
        if self.stack:
            self.stack.pop()

    def handle_data(self, data):
        if not any(t in ("script", "style") for t in self.stack) and data.strip():
            self.found.append(re.sub(r"\s+", " ", data).strip())


def test_memory_is_well_formed():
    mem = memory()
    assert set(mem) == {"words", "de"}
    for part in mem.values():
        assert all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in part.items())
        assert not [k for k, v in part.items() if k == v]


def test_static_texts_of_the_pages_have_german():
    mem = memory()
    for page in ("index.html", "weg2.html"):
        p = Texts()
        with open(os.path.join(STATIC, page), encoding="utf-8") as fh:
            p.feed(fh.read())
        missing = [t for t in p.found if re.search(r"[A-Za-z]{3}", t) and t not in SAME_IN_BOTH and german(t, mem) == t]
        assert not missing, (page, missing)


def test_texts_of_this_change_are_translated():
    mem = memory()
    for en in ("Decode tok/s by bs", "Origin IP", "Context now (tokens)", "Input tokens of the prefill in progress",
               "front delivers no TTFT", "no request answered yet", "no VictoriaMetrics samples yet", "taken over from P (handoff, no cache)",
               "newly prefilled so far (top = whole input)", "Open requests now", "(no session id)"):
        assert german(en, mem) != en, en


def test_page_loads_the_language_modules_in_order():
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
        page = fh.read()
    assert page.index('<script src="i18n.js">') < page.index('<script src="zoom.js">') < page.index('<script src="tooltip.js">') < page.index("<script>\n\"use strict\";")
    for url in ("/i18n.js", "/i18n_de.json", "/tooltip.js"):
        assert url in server.STATIC_FILES and os.path.exists(os.path.join(STATIC, server.STATIC_FILES[url][0]))
    assert 'id="langsw"' in page and 'data-lang="en" class="sel"' in page       # English is the default
