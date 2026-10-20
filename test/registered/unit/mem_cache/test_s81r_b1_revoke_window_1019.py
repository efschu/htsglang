"""S81 review B1 (nf-review-s81-1019 Nachtrag): arena_revoke_orphan_claims must not flip a JOINED slot to EVICTING.

Before B1 the revoke did its CAS CLAIMED -> EVICTING with no look first (unlike free_if_abandoned /
arena_release_claims) and only rolled back to CLAIMED ("kept") after the check under the slot lock. A worker that
had joined TP0's share-0 slot and completed in that window got status 4 ("not CLAIMED": its bytes lost, its writer
left open). Now the same look as the neighbours runs BEFORE the CAS.

Deterministic harness on the REAL arena.c (reviewer template /tmp/rv81c/window.c): the test holds the slot lock, a
thread runs the revoke. Without B1 the thread flips the slot to EVICTING and then waits on the lock -- the harness
sees EVICTING and lets the worker complete inside the window (status 4). With B1 the revoke answers "kept" (1)
without leaving CLAIMED and the worker's completion lands (1).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

from flliper.srt.mem_cache.storage.file import hicache_arena

HARNESS = r'''
#include "arena.c"
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static uint8_t *B; static int64_t SL, GN; static int8_t RST = -1; static _Atomic int DONE = 0;
static void *rev(void *a) { (void)a; arena_revoke_orphan_claims(B, 1, &SL, &GN, &RST); atomic_store(&DONE, 1); return NULL; }
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    int64_t o[6];
    int64_t bytes = arena_layout(4, 4096, o);
    B = aligned_alloc(4096, (size_t)((bytes + 4095) & ~4095LL));
    memset(B, 0, (size_t)bytes);
    if (arena_init(B, 4, 4096) != 0) { puts("init failed"); return 1; }
    uint64_t klo = 77, khi = 99; int64_t tot = 4096, slot = -1, gen = 0; int8_t st = -1;
    arena_claim(B, 1, &klo, &khi, &tot, NULL, &slot, &gen, &st);            /* TP0 fresh claim */
    int64_t zero = 0; int8_t cst = -1;
    arena_complete(B, 1, &slot, &gen, &zero, NULL, NULL, &cst);             /* share 0: open0 */
    int64_t s2 = -1, g2 = 0; int8_t st2 = -1;
    arena_claim(B, 1, &klo, &khi, &tot, NULL, &s2, &g2, &st2);              /* worker JOINS: open1, no bytes */
    SlotHeader *sh = slot_hdr(B, (uint64_t)slot);
    SL = slot; GN = gen;
    atomic_store(&sh->lock, 1);                                             /* hold the slot lock */
    pthread_t t; pthread_create(&t, NULL, rev, NULL);
    int flipped = 0; int8_t wst = -1;
    int64_t one = 1, off = 0, len = 4096;
    double t0 = now_s();
    while (!atomic_load(&DONE) && now_s() - t0 < 2.0) {
        if (atomic_load(&sh->state) == S_EVICTING) {
            flipped = 1;
            arena_complete(B, 1, &s2, &g2, &one, &off, &len, &wst);         /* the worker completes IN the window */
            break;
        }
        sched_yield();
    }
    atomic_store(&sh->lock, 0);
    pthread_join(t, NULL);
    if (!flipped) arena_complete(B, 1, &s2, &g2, &one, &off, &len, &wst);
    printf("join=%d flipped=%d revoke=%d complete=%d state=%u\n", st2, flipped, RST, wst, atomic_load(&sh->state));
    return 0;
}
'''


def _run(tmp_path, arena_src):
    if shutil.which("gcc") is None:
        pytest.skip("no gcc")
    shutil.copy(arena_src, tmp_path / "arena.c")
    (tmp_path / "h.c").write_text(HARNESS)
    exe = tmp_path / "h"
    subprocess.run(["gcc", "-O2", "-pthread", "-o", str(exe), str(tmp_path / "h.c")], check=True,
                   cwd=tmp_path, capture_output=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=30).stdout
    return {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", out)}


def test_b1_joined_slot_is_kept_without_leaving_claimed(tmp_path):
    r = _run(tmp_path, hicache_arena._SRC)
    assert r["join"] == 1, r
    assert r["flipped"] == 0, f"the revoke flipped a joined slot to EVICTING: {r}"
    assert r["revoke"] == 1, f"kept: {r}"
    assert r["complete"] == 1, f"the joiner's completion lands (not 4 'bytes lost'): {r}"
    assert r["state"] == 2, r


def test_b1_orphan_is_still_freed(tmp_path):
    """The look before the CAS keeps the orphan path: a slot of one claimant, none open, is freed."""
    src = HARNESS.replace(
        "    arena_claim(B, 1, &klo, &khi, &tot, NULL, &s2, &g2, &st2);              /* worker JOINS: open1, no bytes */\n",
        "    st2 = 1;\n")
    src = src.replace("    atomic_store(&sh->lock, 1);                                             /* hold the slot lock */\n", "")
    src = src.replace("    if (!flipped) arena_complete(B, 1, &s2, &g2, &one, &off, &len, &wst);\n", "")
    if shutil.which("gcc") is None:
        pytest.skip("no gcc")
    shutil.copy(hicache_arena._SRC, tmp_path / "arena.c")
    (tmp_path / "o.c").write_text(src)
    subprocess.run(["gcc", "-O2", "-pthread", "-o", str(tmp_path / "o"), str(tmp_path / "o.c")], check=True,
                   cwd=tmp_path, capture_output=True)
    out = subprocess.run([str(tmp_path / "o")], check=True, capture_output=True, text=True, timeout=30).stdout
    r = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", out)}
    assert r["revoke"] == 0 and r["state"] == 0, r
