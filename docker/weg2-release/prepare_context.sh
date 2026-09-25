#!/bin/bash
# Build-Kontext fuer das htsglang-Upgrade-Image erzeugen (27B-Sitz R, 24./25.09.2026).
# Liest nur vom Rig und schreibt ausschliesslich unter $OUT; baut kein Image, startet nichts, fasst
# keine GPU an. Waehrend eines Boots unter dem 3-GiB-Deckel laufen lassen (incg.sh, siehe unten).
#
#   ./prepare_context.sh <linie 27b|nf> <commit-sha> <lokaler-branch-der-den-sha-enthaelt> \
#        [--since 2026-09-01] [--with-nv-headers] [--no-tvm-ffi-seed] [--allow-unpushed]
#   ./prepare_context.sh --refresh-jit <ctx-verzeichnis>
#        nach weiteren Rig-Boots: FlashInfer-Modulliste neu aus dem Rig-Cache, tvm-ffi-Saat um neue Eintraege ergaenzen
#   ./prepare_context.sh --refresh-tools <ctx-verzeichnis>
#        nur Dockerfile, tools/ (Entrypoint, Healthcheck, Vorbau, Profile) und .dockerignore neu aus
#        /spinning/gpu-arb/docker uebernehmen -- z.B. wenn ein Profil vom Stand "vorbereitet" auf
#        "abgenommen" wechselt. src/, lock/ und assets/ (die Rig-Messungen) bleiben unberuehrt; der alte
#        Manifest-Digest wird in MANIFEST.history festgehalten, BUILD_INFO.json bekommt den Eintrag.
#
# Ergebnis: /spinning/gpu-arb/docker/ctx/<linie>-<sha10>/ mit Dockerfile, src/ (flacher git-Klon,
# sauber, mit Historie), lock/, assets/, tools/, BUILD_INFO.json, MANIFEST.sha256. Der Proxmox-Host
# sieht es als /spinning/subvol-999-disk-0/spinning/gpu-arb/docker/ctx/... (host_acceptance.sh).
#
# Geheimnisse: nur WHITELIST-Dateien werden kopiert (unten einzeln benannt); zum Schluss ein
# Namens- und Inhalts-Scan, der bei jedem Treffer abbricht. Nie gelesen: PAT-Dateien, *.adminkey,
# gpuq_booking.json, Router-/OpenRouter-Schluessel, /root/.claude/jobs/*.
set -euo pipefail

HERE=/spinning/gpu-arb/docker
say(){ echo "[prepare $(date -u +%H:%M:%SZ)] $*"; }
die(){ echo "!! ABBRUCH: $*" >&2; exit 1; }

secret_scan() {   # secret_scan <ctx>: Namen + Inhalte ausserhalb src/; bricht bei jedem Treffer ab
  local o=$1 hits
  hits=$(find "$o" -path "$o/src" -prune -o \( -name '*.adminkey' -o -name 'gpuq_booking.json' \
          -o -name 'GITHUB_PAT*' -o -name '.git-credentials' -o -name '*.pem' -o -name '.netrc' \) -print)
  [ -z "$hits" ] || die "verbotene Dateien im Kontext: $hits"
  if grep -rIl -E 'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-or-v1-[a-f0-9]{20,}|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|admin[-_]api[-_]key[=": ]+[A-Za-z0-9_-]{20,}' \
       "$o/assets" "$o/lock" "$o/tools" "$o/BUILD_INFO.json" 2>/dev/null; then
    die "Schluessel-Muster im Kontext gefunden (Dateien oben)"
  fi
  if grep -q -E '://[^/@]+:[^/@]+@' "$o/src/.git/config"; then die "Zugangsdaten in src/.git/config"; fi
}
REF_VENV=${REF_VENV:-/spinning/htsglang-gpu/.venv}
fi_modules() {   # fi_modules <fiv>: FlashInfer-Modulliste MIT gencode je Modul aus der Rig-build.ninja (120f und 86) auf stdout
  local fiv=$1 k p u g
  # KEIN Zeitstempel in der Datei: sie ist Cache-Schluessel der Vorbau-Schicht (Dockerfile 3b); der Stand steht in
  # BUILD_INFO.json .jit_snapshot.
  echo "# flashinfer $fiv, Quelle /root/.cache/flashinfer/$fiv/{120f,86}/cached_ops"
  echo "# Spalten: dirkey uri gencode[;gencode] -- 120f traegt compute_120a (Server-Kontext) UND compute_120f (Import-Kontext); jedes Token enthaelt selbst ein Komma"
  for k in 120f 86; do
    for p in /root/.cache/flashinfer/"$fiv"/"$k"/cached_ops/*/; do
      [ -d "$p" ] || continue
      u=$(basename "$p"); [ "$u" = "tmp" ] && continue
      g=$(grep -o -E 'gencode=arch=compute_[0-9a-z]+,code=sm_[0-9a-z]+' "$p/build.ninja" 2>/dev/null | sed 's/^/-/' | sort -u | paste -sd';' -)
      echo "$k $u ${g}"
    done
  done
}
seed_tvm_ffi() {   # seed_tvm_ffi <zielverzeichnis>: vollstaendige Rig-Eintraege (mit .so und Provenienz), fehlende ergaenzen; Anzahl neu
  local dst=$1 d b n=0
  for d in /root/.cache/tvm-ffi/*/; do
    b=$(basename "$d")
    ls "$d"/*.so >/dev/null 2>&1 && [ -f "$d/sgl_jit_provenance.json" ] || continue
    [ -e "$dst/$b" ] && continue
    cp -a "$d" "$dst/$b"; n=$((n + 1))
  done
  echo "$n"
}
write_manifest() {   # write_manifest <ctx>: alles ausser src/ (src/ ist per Commit-SHA und sauberem Baum belegt)
  ( cd "$1" && find Dockerfile assets lock tools BUILD_INFO.json .dockerignore -type f -print0 | sort -z | xargs -0 sha256sum ) > "$1/MANIFEST.sha256"
}

if [ "${1:-}" = "--refresh-tools" ]; then
  OUT=$(cd "${2:?Kontext-Verzeichnis fehlt}" && pwd)
  [ -f "$OUT/MANIFEST.sha256" ] && [ -f "$OUT/BUILD_INFO.json" ] || die "$OUT ist kein fertiger Kontext"
  OLD=$(sha256sum "$OUT/MANIFEST.sha256" | cut -d' ' -f1)
  say "refresh-tools $OUT (Manifest vorher $OLD)"
  CHANGED=0
  for f in Dockerfile .dockerignore; do cmp -s "$HERE/$f" "$OUT/$f" || { say "   geaendert: $f"; CHANGED=1; }; done
  for f in entrypoint.sh healthcheck.sh prebuild_jit.py; do cmp -s "$HERE/$f" "$OUT/tools/$f" || { say "   geaendert: tools/$f"; CHANGED=1; }; done
  for f in "$HERE"/profiles/*.env; do cmp -s "$f" "$OUT/tools/profiles/$(basename "$f")" || { say "   geaendert: tools/profiles/$(basename "$f")"; CHANGED=1; }; done
  for f in "$OUT"/tools/profiles/*.env; do [ -e "$HERE/profiles/$(basename "$f")" ] || { say "   nur im Kontext: tools/profiles/$(basename "$f")"; CHANGED=1; }; done
  if [ "$CHANGED" = 0 ]; then say "refresh-tools: nichts geaendert -- Kontext und Manifest bleiben ($OLD)"; exit 0; fi
  cp -p "$HERE/Dockerfile" "$OUT/Dockerfile"; cp -p "$HERE/.dockerignore" "$OUT/.dockerignore"
  cp -p "$HERE/entrypoint.sh" "$HERE/healthcheck.sh" "$HERE/prebuild_jit.py" "$OUT/tools/"
  cp -p "$HERE"/profiles/*.env "$OUT/tools/profiles/"
  python3 - "$OUT" "$OLD" <<'EOF'
import json, sys, time, pathlib
out, old = sys.argv[1:]
p = pathlib.Path(out, "BUILD_INFO.json"); bi = json.loads(p.read_text())
bi.setdefault("tools_refreshed", []).append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                              "manifest_sha256_before": old})
p.write_text(json.dumps(bi, indent=1))
EOF
  secret_scan "$OUT"
  echo "$(date -u +%FT%TZ) $OLD (vor refresh-tools)" >> "$OUT/MANIFEST.history"
  write_manifest "$OUT"
  say "FERTIG: Manifest jetzt $(sha256sum "$OUT/MANIFEST.sha256" | cut -d' ' -f1)"
  exit 0
fi

if [ "${1:-}" = "--refresh-jit" ]; then
  # Nach weiteren Rig-Boots (neue Formate): FlashInfer-Modulliste neu aus dem Rig-Cache und tvm-ffi-Saat um neue
  # vollstaendige Eintraege ergaenzen. src/, lock/ und die Werkzeuge bleiben; alter Digest -> MANIFEST.history.
  OUT=$(cd "${2:?Kontext-Verzeichnis fehlt}" && pwd)
  [ -f "$OUT/MANIFEST.sha256" ] && [ -f "$OUT/BUILD_INFO.json" ] || die "$OUT ist kein fertiger Kontext"
  OLD=$(sha256sum "$OUT/MANIFEST.sha256" | cut -d' ' -f1)
  FIV=$("$REF_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("flashinfer-python"))')
  fi_modules "$FIV" > "$OUT/assets/flashinfer_modules.txt.new"
  FI_CHANGED=0
  if cmp -s "$OUT/assets/flashinfer_modules.txt" "$OUT/assets/flashinfer_modules.txt.new"; then
    rm -f "$OUT/assets/flashinfer_modules.txt.new"
    say "refresh-jit: FlashInfer-Modulliste byte-gleich ($(grep -c -v '^#' "$OUT/assets/flashinfer_modules.txt") Eintraege)"
  else
    if diff -q <(grep -v '^#' "$OUT/assets/flashinfer_modules.txt") <(grep -v '^#' "$OUT/assets/flashinfer_modules.txt.new") >/dev/null; then
      FI_HEAD_ONLY=1; say "refresh-jit: FlashInfer-Module unveraendert, nur der Kopf (z.B. Zeitstempel entfernt)"
    else
      FI_CHANGED=1; say "refresh-jit: FlashInfer-Modulliste geaendert:"
      diff <(grep -v '^#' "$OUT/assets/flashinfer_modules.txt") <(grep -v '^#' "$OUT/assets/flashinfer_modules.txt.new") | sed 's/^/   /' | head -20
    fi
    mv "$OUT/assets/flashinfer_modules.txt.new" "$OUT/assets/flashinfer_modules.txt"
  fi
  NEW_TVM=$(seed_tvm_ffi "$OUT/assets/tvm-ffi")
  say "refresh-jit: tvm-ffi-Saat +$NEW_TVM Eintraege, jetzt $(find "$OUT/assets/tvm-ffi" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [ "$FI_CHANGED" = 0 ] && [ "$NEW_TVM" = 0 ] && [ "${FI_HEAD_ONLY:-0}" = 0 ]; then
    say "refresh-jit: nichts geaendert -- Kontext und Manifest bleiben ($OLD)"; exit 0
  fi
  python3 - "$OUT" "$OLD" "$FI_CHANGED" "$NEW_TVM" <<'EOF'
import json, sys, time, pathlib
out, old, fi, tv = sys.argv[1:]
p = pathlib.Path(out, "BUILD_INFO.json"); bi = json.loads(p.read_text())
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
bi.setdefault("jit_refreshed", []).append({"utc": now,
    "manifest_sha256_before": old, "flashinfer_modules_changed": fi == "1", "tvm_ffi_added": int(tv)})
mods = [l for l in pathlib.Path(out, "assets/flashinfer_modules.txt").read_text().splitlines() if l and not l.startswith("#")]
bi["jit_snapshot"] = {"utc": now, "flashinfer_modules": len(mods),
                      "tvm_ffi_entries": len([d for d in pathlib.Path(out, "assets/tvm-ffi").iterdir() if d.is_dir()])}
p.write_text(json.dumps(bi, indent=1))
EOF
  secret_scan "$OUT"
  echo "$(date -u +%FT%TZ) $OLD (vor refresh-jit)" >> "$OUT/MANIFEST.history"
  write_manifest "$OUT"
  say "FERTIG: Manifest jetzt $(sha256sum "$OUT/MANIFEST.sha256" | cut -d' ' -f1)"
  exit 0
fi

LINE=${1:?Linie fehlt (27b|nf)}; SHA_IN=${2:?Commit fehlt}; BRANCH=${3:?lokaler Branch fehlt}
shift 3
SINCE=2026-09-01; WITH_NV=0; SEED_TVM=1; ALLOW_UNPUSHED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --since) SINCE=$2; shift 2 ;;
    --allow-unpushed) ALLOW_UNPUSHED=1; shift ;;   # benannte Ausnahme: BUILD_INFO/Image tragen "UNPUSHED", host_publish.sh sperrt
    --with-nv-headers) WITH_NV=1; shift ;;
    --no-tvm-ffi-seed) SEED_TVM=0; shift ;;
    *) echo "unbekannt: $1" >&2; exit 2 ;;
  esac
done
REPO=/spinning/htsglang
ARB=/spinning/gpu-arb
NVSRC=/spinning/nvidia-open-595
WHEEL=/spinning/wt-398-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl
WHEEL_SHA=67f03cfa755efa01498c7732bd6ae015ec5673feffe9a51452fefdbe0dcd4664

case "$LINE" in 27b|nf) ;; *) die "Linie '$LINE' unbekannt";; esac
SHA=$(git -C "$REPO" rev-parse --verify "${SHA_IN}^{commit}") || die "Commit $SHA_IN unbekannt"
git -C "$REPO" merge-base --is-ancestor "$SHA" "$BRANCH" || die "$SHA liegt nicht auf $BRANCH"
# Push-Pruefung: liegt die Revision auf einem Remote-Branch (lokale Remote-Refs, Stand des letzten Fetch)?
# (|| true: grep -v ohne Treffer endet mit 1, unter pipefail beendete das sonst stumm das ganze Skript)
PUSHED_ON=$(git -C "$REPO" branch -r --contains "$SHA" 2>/dev/null | sed 's/^[* ]*//' | { grep -v -- '->' || true; } | head -3 | paste -sd, -)
if [ -n "$PUSHED_ON" ]; then
  PUSH_STATE="pushed: $PUSHED_ON"
elif [ "$ALLOW_UNPUSHED" = "1" ]; then
  PUSH_STATE="UNPUSHED, Veroeffentlichung erst nach Push durch den Nutzer"
  say "!! $PUSH_STATE -- ${SHA:0:10} liegt auf keinem Remote-Branch (benannte Ausnahme --allow-unpushed); BUILD_INFO.json und das Image-Label htsglang.push_state tragen das, host_publish.sh verweigert"
else
  die "${SHA:0:10} liegt auf keinem Remote-Branch (ungepusht) -- erst pushen oder --allow-unpushed (dann sperrt host_publish.sh die Veroeffentlichung)"
fi
# Stufe B (Launcher-Pfadnaehte, das Image setzt SGLANG_WEG2_*): als Vorfahr bae049a3b4 ODER als Pick mit
# gleicher patch-id (RC2-final traegt sie als 68bc631d8c), und in jedem Fall inhaltlich in allen fuenf Dateien.
STAGE_B=""
if git -C "$REPO" merge-base --is-ancestor bae049a3b4 "$SHA"; then STAGE_B="Vorfahr bae049a3b4"
else
  PID_B=$(git -C "$REPO" show bae049a3b4 | git patch-id --stable | cut -d' ' -f1)
  for c in $(git -C "$REPO" rev-list --no-merges --since=2026-09-24 "$SHA" -- python/sglang/srt/weg2/launcher.py); do
    if [ "$(git -C "$REPO" show "$c" | git patch-id --stable | cut -d' ' -f1)" = "$PID_B" ]; then STAGE_B="Pick $c (patch-id = bae049a3b4)"; break; fi
  done
fi
[ -n "$STAGE_B" ] || die "$SHA enthaelt die Stufe-B-Pfadnaehte (bae049a3b4) weder als Vorfahr noch als Pick -- erst picken"
for f in python/sglang/srt/weg2/launcher.py python/sglang/srt/weg2/host_ledger.py python/sglang/srt/weg2/corridor_budget.py \
         scripts/weg2/tms/build_tms_preload.sh test/registered/unit/weg2/test_weg2_rig_paths_env_docker.py; do
  git -C "$REPO" grep -q -E 'SGLANG_WEG2_(GPU_ARB|EVIDENCE_DIR|VENV|TMS_OUT_DIR)' "$SHA" -- "$f" || die "Stufe-B-Naht fehlt inhaltlich in $f @ ${SHA:0:10}"
done
OUT=$HERE/ctx/${LINE}-${SHA:0:10}
[ -e "$OUT" ] && die "$OUT existiert -- nie ueberschreiben (Herkunft muss eindeutig bleiben)"
mkdir -p "$OUT"/{lock,assets/wheels,assets/devtools,assets/arb-seed/weg2/calib,assets/profiles,assets/nvidia-open-595,assets/tvm-ffi,tools/profiles}

say "0/9 $SHA auf $BRANCH, Stufe B: $STAGE_B; Push: $PUSH_STATE"
say "1/9 flacher Klon $BRANCH seit $SINCE -> src/, dann $SHA"
git clone --quiet --no-local --single-branch --branch "$BRANCH" --shallow-since="$SINCE" "file://$REPO" "$OUT/src"
git -C "$OUT/src" -c advice.detachedHead=false checkout --quiet --detach "$SHA"
[ "$(git -C "$OUT/src" rev-parse HEAD)" = "$SHA" ] || die "HEAD im Klon != $SHA"
[ -z "$(git -C "$OUT/src" status --porcelain)" ] || die "Klon nicht sauber"
say "   Historie: $(git -C "$OUT/src" rev-list --count HEAD) Commits (Kalibrier-Identitaet liest git rev-list, line_identity.py:82)"

say "2/9 Lock aus $REF_VENV (pip freeze, ohne sglang/sglang-kernel)"
"$REF_VENV/bin/python" -m pip freeze --exclude-editable > "$OUT/lock/venv-freeze.full.txt"
grep -v -E '^sglang-kernel @ ' "$OUT/lock/venv-freeze.full.txt" > "$OUT/lock/requirements.lock"
if grep -q -E ' @ (file|git)' "$OUT/lock/requirements.lock"; then
  die "Lock enthaelt weitere lokale/VCS-Installationen: $(grep -E ' @ (file|git)' "$OUT/lock/requirements.lock" | cut -c1-80)"
fi
say "   $(wc -l < "$OUT/lock/requirements.lock") Pakete, $(grep -c -i cu12 "$OUT/lock/requirements.lock") cu12-Reste (bewusst behalten)"

say "3/9 Kernel-Wheel (#384-Pin)"
echo "$WHEEL_SHA  $WHEEL" | sha256sum -c --quiet - || die "Wheel-Hash weicht vom Pin ab"
cp -p "$WHEEL" "$OUT/assets/wheels/"

DRV=$(awk '/NVRM version/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/){print $i; exit}}' /proc/driver/nvidia/version)
NVDESC=none; NVDIFF=none
if [ "$WITH_NV" = "1" ]; then
  say "4/9 NV-Header (optional, 3 Verzeichnisse des gepatchten Baums $NVSRC)"
  for d in kernel-open/common/inc src/common/sdk/nvidia/inc src/nvidia/arch/nvalloc/unix/include; do
    mkdir -p "$OUT/assets/nvidia-open-595/$(dirname "$d")"
    cp -a "$NVSRC/$d" "$OUT/assets/nvidia-open-595/$d"
  done
  NVDESC=$(git -C "$NVSRC" describe --tags --always 2>/dev/null || echo unknown)
  NVDIFF=$(git -C "$NVSRC" diff -- kernel-open/common/inc src/common/sdk/nvidia/inc src/nvidia/arch/nvalloc/unix/include | sha256sum | cut -c1-16)
  { echo "tree=$NVSRC describe=$NVDESC header_patch_sha256_16=$NVDIFF host_driver=$DRV"
    git -C "$NVSRC" diff --stat -- kernel-open/common/inc src/common/sdk/nvidia/inc src/nvidia/arch/nvalloc/unix/include; } \
    > "$OUT/assets/nvidia-open-595/PROVENANCE.txt"
else
  say "4/9 NV-Header: AUS (Default, --with-nv-headers setzt sie ins Image)"
  echo "leer: Image ohne NV-Header gebaut (WITH_NV_HEADERS=0)" > "$OUT/assets/nvidia-open-595/.keep"
fi

say "5/9 Rig-Werkzeuge (Whitelist), ARB-Saat, Profil-Daten ($LINE)"
for f in boot_deadman.sh host_ledger_preflight.sh mem_timeseries.sh; do
  cp -p "$ARB/devtools/$f" "$OUT/assets/devtools/$f"
done
cp -p "$ARB/weg2/PROBE_RING_0907.md" "$OUT/assets/arb-seed/weg2/"
cp -p "$ARB"/weg2/calib/*.json "$OUT/assets/arb-seed/weg2/calib/"
if [ "$LINE" = "27b" ]; then
  cp -p "$ARB/weg2/corridor_budget_sample.json" "$OUT/assets/arb-seed/weg2/"      # Launcher-Default-Pfad (kein Flag im 27B-Arm)
  mkdir -p "$OUT/assets/profiles/27b"
  cp -p "$ARB/weg2/census/xchg_census_weg2xsn246_27198a2711.json" "$OUT/assets/profiles/27b/"
else
  mkdir -p "$OUT/assets/profiles/nf"                                               # NF_PROFILE.md §11.1/§11.3 N4
  cp -p "$ARB/weg2/census/xchg_census_fnFL2_graph.json" "$ARB/weg2/census/xchg_census_fnFL2_computed.json" \
        "$ARB/weg2/corridor_budget_sample_nextflash_0921.json" "$OUT/assets/profiles/nf/"
fi
# Weitere Profil-Daten je Linie aus profiles/<linie>.assets (eine Rig-Datei je Zeile, # = Kommentar) nach
# assets/profiles/<linie>/ -- im Image /opt/htsglang/profiles/<linie>/<name>. Fehlt eine Datei: Abbruch.
if [ -f "$HERE/profiles/$LINE.assets" ]; then
  mkdir -p "$OUT/assets/profiles/$LINE"
  while IFS= read -r a; do
    a=${a%%#*}; a=$(echo "$a" | xargs); [ -n "$a" ] || continue
    [ -f "$a" ] || die "Profil-Datei fehlt: $a (profiles/$LINE.assets)"
    cp -p "$a" "$OUT/assets/profiles/$LINE/"
  done < "$HERE/profiles/$LINE.assets"
  say "   Profil-Daten aus profiles/$LINE.assets: $(ls "$OUT/assets/profiles/$LINE" | wc -l) Dateien, $(du -sh --apparent-size "$OUT/assets/profiles/$LINE" | cut -f1)"
fi

say "6/9 FlashInfer-Modulliste MIT gencode je Modul aus der Rig-build.ninja (120f und 86)"
FIV=$("$REF_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("flashinfer-python"))')
fi_modules "$FIV" > "$OUT/assets/flashinfer_modules.txt"
say "   $(grep -c -v '^#' "$OUT/assets/flashinfer_modules.txt") Modul-Eintraege"

if [ "$SEED_TVM" = "1" ]; then
  say "7/9 tvm-ffi-Saat (inhaltsadressiert; nur vollstaendige Eintraege mit .so und Provenienz)"
  n=$(seed_tvm_ffi "$OUT/assets/tvm-ffi")
  say "   $n Eintraege, $(du -sh "$OUT/assets/tvm-ffi" | cut -f1)"
else
  say "7/9 tvm-ffi-Saat: AUS"; echo "leer" > "$OUT/assets/tvm-ffi/.keep"
fi

say "8/9 Dockerfile, Werkzeuge, Profile, .dockerignore, BUILD_INFO.json"
cp -p "$HERE/Dockerfile" "$OUT/Dockerfile"
cp -p "$HERE/entrypoint.sh" "$HERE/healthcheck.sh" "$HERE/prebuild_jit.py" "$OUT/tools/"
cp -p "$HERE"/profiles/*.env "$OUT/tools/profiles/"
cp -p "$HERE/.dockerignore" "$OUT/.dockerignore"
python3 - "$OUT" "$LINE" "$SHA" "$BRANCH" "$SINCE" "$REF_VENV" "$DRV" "$NVDESC" "$NVDIFF" "$FIV" "$WHEEL_SHA" "$WITH_NV" "$SEED_TVM" "$STAGE_B" "$PUSH_STATE" <<'EOF'
import hashlib, json, sys, time, pathlib
out, line, sha, branch, since, venv, drv, nvdesc, nvdiff, fiv, wsha, with_nv, seed, stage_b, push_state = sys.argv[1:]
lock = pathlib.Path(out, "lock/requirements.lock").read_bytes()
# F10: welche NCCL die Linie am Rig WIRKLICH laedt -- die Datei, ihr Banner, und welcher RECORD sie beansprucht
# (nvidia-nccl-cu12 und -cu13 listen beide nvidia/nccl/lib/libnccl.so.2; der Hash sagt, wessen Inhalt dort liegt).
import base64, glob, os, re
sp = pathlib.Path(venv, "lib/python3.12/site-packages")
lib = sp / "nvidia/nccl/lib/libnccl.so.2"
data = lib.read_bytes()
m = re.search(rb"NCCL version [0-9.]+\+cuda[0-9.]+", data)
rec_hash = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
claims = {}
for rec in sorted(glob.glob(str(sp / "nvidia_nccl_cu1*.dist-info/RECORD"))):
    for row in open(rec):
        if row.startswith("nvidia/nccl/lib/libnccl.so.2,"):
            claims[os.path.basename(os.path.dirname(rec))] = row.split(",")[1] == rec_hash
nccl = {"file": str(lib), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
        "banner": m.group(0).decode() if m else None, "record_claims_match": claims}
json.dump({
    "line": line, "revision": sha, "branch": branch, "shallow_since": since,
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "reference_venv": venv, "lock_sha256": hashlib.sha256(lock).hexdigest(),
    "kernel_wheel_sha256": wsha, "flashinfer": fiv, "driver_expected": drv,
    "nv_headers": {"in_image": with_nv == "1", "describe": nvdesc, "patch_sha256_16": nvdiff},
    "tvm_ffi_seed": seed == "1",
    "stage_b": stage_b,
    "push_state": push_state,
    "nccl": nccl,
    "base_image": "beim Build per --build-arg CUDA_BASE (Digest) festschreiben",
    "lineage": {"published": "ghcr.io/efschu/htsglang:cu130-nccl2307 (2026-07-14)",
                "august": "chore/release-chain-prep-r3 @ ab4a42d392, htsglang:r2-99a4b0a4-gated (2026-08-14)"},
}, open(pathlib.Path(out, "BUILD_INFO.json"), "w"), indent=1)
EOF

python3 - "$OUT" <<'EOF'
import json, sys, time, pathlib
out = pathlib.Path(sys.argv[1]); p = out / "BUILD_INFO.json"; bi = json.loads(p.read_text())
mods = [l for l in (out / "assets/flashinfer_modules.txt").read_text().splitlines() if l and not l.startswith("#")]
bi["jit_snapshot"] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "flashinfer_modules": len(mods),
                      "tvm_ffi_entries": len([d for d in (out / "assets/tvm-ffi").iterdir() if d.is_dir()])}
p.write_text(json.dumps(bi, indent=1))
EOF
say "9/9 Geheimnis-Scan (Namen + Inhalte ausserhalb src/) und Manifest"
secret_scan "$OUT"
write_manifest "$OUT"
say "FERTIG: $OUT ($(du -sh "$OUT" | cut -f1)); Manifest $(sha256sum "$OUT/MANIFEST.sha256" | cut -d' ' -f1); Build nur nach Go des Operators."
