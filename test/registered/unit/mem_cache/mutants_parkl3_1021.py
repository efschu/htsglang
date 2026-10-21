"""PARK-L3 mutants (S86e, #59c, W3-ROOM-SET EMPTY): each weakens one rule along a named danger direction; the named
test file must go RED on every one. Serial (one pytest at a time), under MemoryMax=2G, no GPU. A mutant string must
match exactly once, else the mutant is INVALID (not SURVIVED).

  python test/registered/unit/mem_cache/mutants_parkl3_1021.py  (from the worktree root)"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
R = os.path.join(ROOT, "python/flliper/srt/")
T = "test/registered/unit/mem_cache/"
S86E = T + "test_s86e_park_l3_host_only_1021.py"
P59C = T + "test_59c_park_secured_depth_1021.py"
EMPTY = T + "test_w3_room_set_empty_census_1021.py"

M = [
    # danger 1: a depth reported beyond the secured pages
    ("D1a every node counts as secured", R + "managers/pdflip_resumable_depth.py",
     "    if bool(getattr(node, \"l3_present\", False)):\n        return True\n",
     "    return True\n", P59C),
    ("D1b the answer is the admission depth", R + "managers/pdflip_resumable_depth.py",
     "ans = {rid: max(0, min(int(d), int(c))) for rid, d, c in zip(rids, depths, caps)}",
     "ans = {rid: max(0, int(d)) for rid, d, c in zip(rids, depths, caps)}", P59C),
    ("D1c the cap is not reduced with the depth", R + "managers/pdflip_resumable_depth.py",
     "            both = list((reduce_min or _tp_min)(depths + caps))\n            depths, caps = both[:len(reqs)], both[len(reqs):]",
     "            depths = list((reduce_min or _tp_min)(depths + caps))[:len(reqs)]", P59C),
    ("D1d the read clamp takes the capped answer", R + "pdflip/d_park_runtime.py",
     "                req, resumable_raw.get(str(req.rid)), is_bigram=bigram\n            ) is not None:\n"
     "                logger.info(\"PDFLIP-D-PARK READ=RESUMABLE rid=%s cap %s -> %s\",\n"
     "                            str(req.rid), before, d_park_read.describe(req))\n    _ph(\"clamp\")",
     "                req, resumable.get(str(req.rid)), is_bigram=bigram\n            ) is not None:\n"
     "                logger.info(\"PDFLIP-D-PARK READ=RESUMABLE rid=%s cap %s -> %s\",\n"
     "                            str(req.rid), before, d_park_read.describe(req))\n    _ph(\"clamp\")", P59C),
    # danger 2: the sleep frees an unsecured page silently
    ("D2a host-only node released without its L3 proof", R + "pdflip/sleep_l3_release.py",
     "if r is None and sl and not _l3_proven(pool, node):",
     "if r is None and sl and not host_only and not _l3_proven(pool, node):", S86E),
    ("D2b a host-only parent leaves before its kept child", R + "pdflip/sleep_l3_release.py",
     "                else:\n                    bad = True\n", "                else:\n                    pass\n", S86E),
    ("D2c the subtree's vetoed slots do not block the parent", R + "pdflip/sleep_l3_release.py",
     "            sub_vetoed = any(int(agreed[s]) for s in sub)", "            sub_vetoed = False", S86E),
    ("D2d the anchor is never judged", R + "pdflip/sleep_l3_release.py",
     "    r = _fls._secure_child_anchors(tree, [node], [_fls.ANCHOR_SECURE_PER_FLUSH])\n    if r is None:",
     "    r = None\n    if r is None:", S86E),
    ("D2e host rows outside the arena removed", R + "pdflip/sleep_l3_release.py",
     "            if r is None and host_only and not sl:\n", "            if False:\n", S86E),
    # (an "apply checks the node's own slots, not its subtree's" mutant is EQUIVALENT at the fixed point: a host-only
    # node with a vetoed subtree slot has all its own slots vetoed by _mixed_block -- the apply check is depth)
    # danger 3: an empty W3 decision stays silent
    ("D3a the empty decision is not noted", R + "pdflip/w3_room_set.py",
     "            _note_empty(cache, pool, want, ctx)   # PARK-L3", "            pass   # PARK-L3", EMPTY),
    ("D3b running requests not named", R + "pdflip/w3_room_set.py",
     "        return \"told_head\"\n    cd = node.component_data[BASE_COMPONENT_TYPE]\n"
     "    if int(getattr(cd, \"lock_ref\", 0) or 0) > 0:\n        return \"running\"\n",
     "        return \"told_head\"\n    cd = node.component_data[BASE_COMPONENT_TYPE]\n", EMPTY),
]


def run(test):
    cmd = ["systemd-run", "--scope", "-q", "-p", "MemoryMax=2G", "env", "CUDA_VISIBLE_DEVICES=",
           "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=python:" + T, "/spinning/shvllm/.venv/bin/python", "-m", "pytest",
           "-q", "-x", "-p", "no:cacheprovider", test]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).returncode


def main():
    killed = invalid = survived = 0
    for name, path, old, new, test in M:
        src = open(path).read()
        if src.count(old) != 1:
            print("INVALID  %s (matches %d)" % (name, src.count(old)))
            invalid += 1
            continue
        open(path, "w").write(src.replace(old, new, 1))
        try:
            rc = run(test)
        finally:
            open(path, "w").write(src)
        if rc != 0:
            print("KILLED   %s" % name)
            killed += 1
        else:
            print("SURVIVED %s" % name)
            survived += 1
    print("killed=%d survived=%d invalid=%d of %d" % (killed, survived, invalid, len(M)))
    return 1 if (survived or invalid) else 0


if __name__ == "__main__":
    sys.exit(main())
