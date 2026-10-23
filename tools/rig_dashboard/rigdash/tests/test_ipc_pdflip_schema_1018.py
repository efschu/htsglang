"""fLLiper (renamed tree) writes the IPC records as pdflip.*/1; the dashboard reads both lines (18.10.)."""
import json
import os
import time

from rigdash import ipcfields, ipcstate


def _boot(root, name, schema):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "state.json"), "w") as f:
        json.dump({"schema": schema, "tag": name, "lifecycle": {"state": "serving"}}, f)
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        f.write(json.dumps({"schema": schema.replace("state", "event"), "kind": "group_ready", "ts": time.time()}) + "\n")
    return d


def test_pdflip_and_old_line_boots_are_both_read(tmp_path):
    root = str(tmp_path)
    _boot(root, "old-boot", "we" "g2.state/1")   # the OLD line's schema, spelled in two parts so the rename tool leaves it (the test is about reading it)
    _boot(root, "fl-boot", "pdflip.state/1")
    _boot(root, "alien-boot", "other.state/1")
    st = ipcstate.IpcStates(roots=(root,))
    st.poll()
    names = sorted(os.path.basename(d) for d in st._st)
    assert names == ["fl-boot", "old-boot"]


def test_rankstats_schema_accepts_pdflip():
    assert "pdflip.rankstats/1" in ipcfields.RANKSTATS_SCHEMAS
    assert ipcfields.RANKSTATS_SCHEMA in ipcfields.RANKSTATS_SCHEMAS
