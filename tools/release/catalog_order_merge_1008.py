#!/usr/bin/env python3
"""catalog_order_merge_1008.py <generated.json> <reference catalog.json> <out.json>   (F0-I)

Content = the generator's output (profile_catalog.py), key order = the reference file (new keys go behind their predecessor in the
generator's order), so a rebuilt catalog.json shows content changes in the diff, not key shuffling.  Written as the shipped file:
json.dumps(indent=1, ensure_ascii=False) + newline.  Exits 1 if the content is not exactly the generator's.
"""
import json
import sys

gen_p, ref_p, out_p = sys.argv[1:4]
gen = json.load(open(gen_p, encoding="utf-8"))
ref = json.load(open(ref_p, encoding="utf-8"))


def order(g, r):
    if isinstance(g, dict) and isinstance(r, dict):
        keys = [k for k in r if k in g]
        gk = list(g)
        for i, k in enumerate(gk):
            if k in r:
                continue
            pred = gk[i - 1] if i else None
            keys.insert(keys.index(pred) + 1 if pred in keys else 0, k)
        return {k: order(g[k], r.get(k)) for k in keys}
    return g


res = order(gen, ref)
if json.dumps(res, sort_keys=True) != json.dumps(gen, sort_keys=True):
    sys.exit(1)
with open(out_p, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
