#!/bin/bash
# STAGED (item 600): host_publish_flliper_selftest.sh.new_600, derived from the live file sha256 6d8bd5b6e96d798b7fa0f6504eadcf3c2e8234c4abcfc8b525d20739e19a58fa (Duo cases)
# host_publish_flliper_selftest.sh -- self-test of host_publish_flliper.sh with a FAKE docker (PATH shim). No real docker,
# no network, no registry, no login. PB for the 27B operator, 2026-09-26.
#
#   bash /spinning/gpu-arb/docker/host_publish_flliper_selftest.sh [-v]
#
# Everything lives in one mktemp dir under /tmp (removed at the end; -v keeps it and prints every case's output):
#   * bin/docker        the shim: image inspect / history / manifest inspect / tag / push / run from a state dir; every
#                       call is logged; `login` is refused and flagged; `run` must carry --rm --network none --read-only
#                       and then executes the gate's REAL in-container scan script with /bin/sh on a fixture tree
#                       (its /tmp/ paths rewritten into the case dir);
#   * remote.git + repo git fixture: c0 python/sglang only (pushed, branch old), c1 python/flliper (pushed, main),
#                       c2 python/flliper (local branch only, never pushed);
#   * per case a fresh state dir (labels, env, history, tags, registry, verdicts, Go file, scan tree).
# Case = mutation of the all-green baseline; the case passes when the set of RED locks equals the expected set
# (and rc / tag / push calls match). Every lock is driven red at least once and green in the baseline.
# Fake tokens are assembled at runtime, so this file carries no token-shaped string.
set -uo pipefail
VERBOSE=0; [ "${1:-}" = -v ] && VERBOSE=1
HERE=$(cd "$(dirname "$0")" && pwd)
GATE=$HERE/host_publish_flliper.sh
[ -f "$GATE" ] || { echo "gate $GATE missing"; exit 2; }
if grep -nE '(^|[^A-Za-z_./-])/[a-z/]*bin/docker' "$GATE"; then echo "gate calls docker by absolute path -- the shim could be bypassed"; exit 2; fi
T=$(mktemp -d /tmp/publish_flliper_selftest.XXXXXX) || exit 2
[ "$VERBOSE" = 1 ] && echo "selftest dir: $T" || trap 'rm -rf "$T"' EXIT
export HOME=$T/home GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0
mkdir -p "$HOME" "$T/bin"

# ---- fake docker ------------------------------------------------------------------------------------------------------
cat > "$T/bin/docker" <<'SHIM'
#!/bin/bash
F=${FAKE_DOCKER_STATE:?FAKE_DOCKER_STATE}
printf '%s\n' "$*" >> "$F/calls.log"
id=$(cat "$F/id")
resolve(){ [ "$1" = "$id" ] && { echo "$id"; return 0; }; awk -v r="$1" '$1==r{print $2; f=1} END{exit !f}' "$F/tags"; }
case "$1" in
  login) echo LOGIN-ATTEMPT >> "$F/calls.log"; exit 97 ;;
  image)
    [ "$2" = inspect ] || exit 98
    shift 2; fmt=""; [ "$1" = -f ] && { fmt=$2; shift 2; }
    rid=$(resolve "$1") || exit 1
    [ "$rid" = "$id" ] || { [ "$fmt" = '{{.Id}}' ] && { echo "$rid"; exit 0; }; exit 1; }
    case "$fmt" in
      *'{{$k}}='*) cat "$F/labels.kv" ;;
      *Config.Labels*) cut -d= -f2- "$F/labels.kv" ;;
      *Config.Env*) cat "$F/env.txt" ;;
      *RepoDigests*) echo "fake/flliper@sha256:$(printf %064d 7)" ;;
      '{{.Id}}') echo "$id" ;;
      *) exit 98 ;;
    esac ;;
  history) [ -f "$F/history_fail" ] && exit 1; cat "$F/history.txt" ;;
  manifest) [ "$2" = inspect ] && grep -qxF "$3" "$F/registry" && exit 0; exit 1 ;;
  tag) printf '%s %s\n' "$3" "$2" >> "$F/tags"; echo "TAG $3" >> "$F/actions" ;;
  push) echo "PUSH $2" >> "$F/actions" ;;
  run)
    shift; rm=0; net=""; ro=0; img=""
    while [ $# -gt 0 ]; do
      case "$1" in
        --rm) rm=1; shift ;; --read-only) ro=1; shift ;; --network) net=$2; shift 2 ;;
        -e) export "${2%%=*}=${2#*=}"; shift 2 ;;
        --tmpfs|--user|--entrypoint) shift 2 ;;
        -*) echo "shim: unexpected run option $1" >&2; exit 96 ;;
        *) img=$1; shift; break ;;
      esac
    done
    [ "$rm" = 1 ] && [ "$net" = none ] && [ "$ro" = 1 ] || { echo "RUN-UNSAFE" >> "$F/calls.log"; exit 95; }
    [ "$img" = "$id" ] || exit 1
    [ -f "$F/run_fail" ] && exit 125
    [ "$1" = -c ] || exit 94
    mkdir -p "$F/ctmp"; script=${2//\/tmp\//$F/ctmp/}
    out=$(/bin/sh -c "$script")
    [ -f "$F/run_cut" ] && out=$(printf '%s\n' "$out" | grep -v SCAN-END)
    printf '%s\n' "$out" ;;
  *) exit 98 ;;
esac
SHIM
chmod +x "$T/bin/docker"
export PATH=$T/bin:$PATH
[ "$(command -v docker)" = "$T/bin/docker" ] || { echo "PATH shim not first -- refusing (would reach real docker)"; exit 2; }

# ---- git fixture ------------------------------------------------------------------------------------------------------
g(){ git -c user.name=selftest -c user.email=selftest@invalid -c init.defaultBranch=main "$@" >/dev/null 2>&1; }
g init --bare "$T/remote.git"; g clone "$T/remote.git" "$T/repo"
R=$T/repo
mkdir -p "$R/python/sglang"; echo x > "$R/python/sglang/__init__.py"
g -C "$R" add -A; g -C "$R" commit -m c0; g -C "$R" branch old; g -C "$R" push origin old
C0=$(git -C "$R" rev-parse HEAD)
mkdir -p "$R/python/flliper"; echo y > "$R/python/flliper/__init__.py"
g -C "$R" add -A; g -C "$R" commit -m c1; g -C "$R" push origin main
C1=$(git -C "$R" rev-parse HEAD)
g -C "$R" checkout -b local-only; echo z >> "$R/python/flliper/__init__.py"; g -C "$R" commit -am c2
C2=$(git -C "$R" rev-parse HEAD)
g -C "$R" checkout main; g -C "$R" fetch origin
echo w >> "$R/python/flliper/__init__.py"; g -C "$R" commit -am c3; g -C "$R" push origin main
C3=$(git -C "$R" rev-parse HEAD)
[ -n "$C0" ] && [ -n "$C1" ] && [ -n "$C2" ] && [ -n "$C3" ] || { echo "git fixture failed"; exit 2; }

ID=sha256:$(printf 'ab%.0s' {1..32})
OTHER=sha256:$(printf 'cd%.0s' {1..32})
HEX=${ID#sha256:}
TOK_GH="gh""p_$(printf 'Q%.0s' {1..18})$(printf 'w%.0s' {1..18})"
TOK_HF="h""f_$(printf 'Z%.0s' {1..17})$(printf 'k%.0s' {1..17})"
PASSC=0; FAILC=0; FAILED=()

# base <dir> <rev>: write the all-green state for revision <rev>
base(){ local F=$1 rev=$2 v=0.1.0-rc1
  mkdir -p "$F/v27" "$F/nf" "$F/go" "$F/img/opt/flliper" "$F/img/etc" "$F/img/root"
  echo "$ID" > "$F/id"; echo "flliper-cand:test $ID" > "$F/tags"; : > "$F/registry"; : > "$F/calls.log"; : > "$F/actions"
  cat > "$F/labels.kv" <<EOF
org.opencontainers.image.title=fLLiper
org.opencontainers.image.version=$v
org.opencontainers.image.revision=$rev
org.opencontainers.image.source=https://github.com/efschu/fLLiper
org.opencontainers.image.licenses=Apache-2.0
org.opencontainers.image.description=fLLiper: prefill/decode flip serving
org.opencontainers.image.base.name=nvidia/cuda@sha256:$(printf %064d 1)
io.github.efschu.flliper.slots=27b nf
io.github.efschu.flliper.revision=$rev
io.github.efschu.flliper.revision.27b=$rev
io.github.efschu.flliper.revision.nf=$rev
io.github.efschu.flliper.push_state=pushed: origin/main
io.github.efschu.flliper.cuda=cu130
io.github.efschu.flliper.kernel_wheel.sha256=$(printf %064d 2)
io.github.efschu.flliper.kernel_archs=86;120a
io.github.efschu.flliper.flashinfer=0.7.0+2f3bc5ac
io.github.efschu.flliper.cute_dsl=4.7.1
io.github.efschu.flliper.nccl=2.28.9+cuda13.0
io.github.efschu.flliper.driver.expected=595.58.03
io.github.efschu.flliper.lock.sha256=$(printf %064d 3)
io.github.efschu.flliper.build=flat
io.github.efschu.flliper.placeholders=none
EOF
  printf 'PATH=/usr/bin\nFLLIPER_PROFILE=27b\nHF_TOKEN=\n' > "$F/env.txt"
  printf '/bin/sh -c #(nop) ENTRYPOINT ["/opt/flliper/entrypoint.sh"]\nRUN |2 KERNEL_WHEEL_SHA256=%064d FLLIPER_VERSION=%s /bin/sh -c pip install\n' 2 "$v" > "$F/history.txt"
  printf 'print("hello")\nurl = "https://github.com/efschu/fLLiper"\nkey = "AKIAIOSFODNN7EXAMPLE"\n' > "$F/img/opt/flliper/app.py"
  printf 'root:x:0:0::/root:/bin/sh\n' > "$F/img/etc/passwd"
  printf '# 27B INT8 final acceptance -- verdict (RUNID r1)\n\nImage `flliper-cand:test` (release %s, 27B tree `%s`), profile `27b-rc10`, boot `x`. Overall: **PASS**\n\n| 1 | a | b | c | PASS | d |\n' "$v" "${rev:0:10}" > "$F/v27/verdict_r1.md"
  printf 'image=flliper-cand:test\nimage_id=%s\n' "$ID" > "$F/v27/facts_r1.kv"
  printf '# NF INT4 final acceptance -- verdict\n\nImage id %s, tree `%s`. Overall: **PASS**\n' "$ID" "${rev:0:10}" > "$F/nf/verdict_nf.md"
  touch -d '2026-09-26 18:00' "$F/v27/verdict_r1.md" "$F/nf/verdict_nf.md"
  printf 'digest: %s\ngo: "ja, veroeffentlichen" (user, verbatim, 2026-09-26T19:00Z, via operator)\n' "$ID" > "$F/go/$HEX.go"
  touch -d '2026-09-26 19:00' "$F/go/$HEX.go"
}
setlabel(){ local F=$1 k=$2 v=$3; grep -v "^$k=" "$F/labels.kv" > "$F/l.tmp"; [ "$v" = __DEL__ ] || echo "$k=$v" >> "$F/l.tmp"; mv "$F/l.tmp" "$F/labels.kv"; }

# case <name> <expected red set, e.g. "1" / "2 6" / "-"> <expected rc> <mutation (eval'd, F set)> [gate args...]
case_(){ local name=$1 want=$2 wantrc=$3 mut=$4; shift 4
  local F=$T/case_$((PASSC + FAILC + 1)) rev=$C1 out rc got act
  mkdir -p "$F"; REVC=$C1; base "$F" "$C1"
  REGISTRY_ARG=(--registry ghcr.io/efschu/flliper); REMOTE_CHECK=tracking
  eval "$mut"
  out=$(env -u REGISTRY FAKE_DOCKER_STATE="$F" S_ROOT="" REPO="$R" REMOTE=origin REMOTE_CHECK="$REMOTE_CHECK" \
        VERDICT_27B_DIR="$F/v27" NF_VERDICT="${NFV-$F/nf/verdict_nf.md}" GO_DIR="$F/go" IMAGE=flliper-cand:test \
        SCAN_PATHS="$F/img/opt $F/img/etc $F/img/root" bash "$GATE" "${REGISTRY_ARG[@]}" "$@" 2>&1); rc=$?
  unset NFV
  got=$(printf '%s\n' "$out" | awk '/^   [1-6] [a-z-]+: RED/{printf "%s ", $1}' | sed 's/ $//'); [ -n "$got" ] || got=-
  act=$(paste -sd' ' "$F/actions")
  local okc=1 why=""
  [ "$got" = "$want" ] || { okc=0; why="red locks '$got' != '$want'"; }
  [ "$rc" = "$wantrc" ] || { okc=0; why="$why rc $rc != $wantrc"; }
  grep -q LOGIN-ATTEMPT "$F/calls.log" && { okc=0; why="$why LOGIN attempted"; }
  grep -q RUN-UNSAFE "$F/calls.log" && { okc=0; why="$why unsafe docker run"; }
  if [ "${EXPECT_PUSH:-0}" = 1 ]; then
    [ "$act" = "${WANT_ACT:-}" ] || { okc=0; why="$why actions '$act' != '${WANT_ACT:-}'"; }
  else
    [ -z "$act" ] || { okc=0; why="$why tag/push happened: $act"; }
  fi
  if printf '%s\n' "$out" | grep -qF -e "$TOK_GH" -e "$TOK_HF"; then okc=0; why="$why TOKEN PRINTED"; fi
  if [ "$okc" = 1 ]; then PASSC=$((PASSC + 1)); printf 'PASS  %-58s red=%s rc=%s\n' "$name" "$got" "$rc"
  else FAILC=$((FAILC + 1)); FAILED+=("$name"); printf 'FAIL  %-58s %s\n' "$name" "$why"; printf '%s\n' "$out" | sed 's/^/      | /'; fi
  [ "$VERBOSE" = 1 ] && [ "$okc" = 1 ] && printf '%s\n' "$out" | sed 's/^/      | /'
  EXPECT_PUSH=0; WANT_ACT=""
}

echo "== host_publish_flliper.sh self-test (fake docker $T/bin/docker; git fixture c0=${C0:0:10} c1=${C1:0:10} c2=${C2:0:10} c3=${C3:0:10})"
# ---- all green ---------------------------------------------------------------------------------------------------------
case_ "baseline plan --scan: all six green"                    -  0 ':' --scan
case_ "baseline, REMOTE_CHECK=ls-remote"                       -  0 'REMOTE_CHECK=ls-remote' --scan
case_ "baseline, 27B id in verdict instead of facts"           -  0 'rm "$F/v27/facts_r1.kv"; echo "image id $ID" >> "$F/v27/verdict_r1.md"' --scan
# ---- all red at once ----------------------------------------------------------------------------------------------------
case_ "ALL RED: duo label, B4 placeholders, tag-only verdict, no Go, token, probe version" "1 2 3 4 5 6" 3 '
  setlabel "$F" org.opencontainers.image.revision "27b=${C1:0:10} nf=${C1:0:10}"
  setlabel "$F" io.github.efschu.flliper.placeholders "source (B4, user decision pending)"
  setlabel "$F" org.opencontainers.image.source UNDECIDED-B4-source-repo
  setlabel "$F" org.opencontainers.image.version 0.0.0-probe1
  rm "$F/v27/facts_r1.kv" "$F/go/$HEX.go"; echo "t = \"$TOK_GH\"" >> "$F/img/root/notes.txt"' --publish
# ---- lock 1 -------------------------------------------------------------------------------------------------------------
case_ "L1 placeholders label names open B4 value"               1  3 'setlabel "$F" io.github.efschu.flliper.placeholders "version (B4, user decision pending)"' --scan
case_ "L1 placeholders label absent"                            1  3 'setlabel "$F" io.github.efschu.flliper.placeholders __DEL__' --scan
case_ "L1 required label missing (cute_dsl)"                    1  3 'setlabel "$F" io.github.efschu.flliper.cute_dsl __DEL__' --scan
case_ "L1 leftover htsglang.* label"                            1  3 'echo "htsglang.line=27b" >> "$F/labels.kv"' --scan
case_ "L1 build label not flat"                                 1  3 'setlabel "$F" io.github.efschu.flliper.build delta' --scan
case_ "L1 lock.sha256 short"                                    1  3 'setlabel "$F" io.github.efschu.flliper.lock.sha256 ae10b55f' --scan
# ---- lock 2 -------------------------------------------------------------------------------------------------------------
case_ "L2 duo revision label (old Dockerfile form)"        "2 3 6" 3 'setlabel "$F" org.opencontainers.image.revision "27b=$C1 nf=$C1"' --scan
case_ "L2 revision.nf is an UNRENAMED other tree (Duo)"         "2 3" 3 'setlabel "$F" io.github.efschu.flliper.revision.nf $C0' --scan
case_ "L2 revision.nf is not a 40-hex SHA"                      "2 3" 3 'setlabel "$F" io.github.efschu.flliper.revision.nf nf-tree' --scan
case_ "L2 revision.27b differs from the OCI revision"           2  3 'setlabel "$F" io.github.efschu.flliper.revision.27b $C3' --scan
case_ "DUO green: nf = other renamed+pushed tree, NF verdict names it" - 0 'setlabel "$F" io.github.efschu.flliper.revision.nf $C3; sed -i "s/tree \`[0-9a-f]*\`/tree \`${C3:0:10}\`/" "$F/nf/verdict_nf.md"; touch -d "2026-09-26 18:00" "$F/nf/verdict_nf.md"' --scan
case_ "DUO: nf = other renamed tree, NF verdict still names the 27B tree" 3 3 'setlabel "$F" io.github.efschu.flliper.revision.nf $C3' --scan
case_ "DUO: nf = renamed but UNPUSHED tree (c2)"                "2 3" 3 'setlabel "$F" io.github.efschu.flliper.revision.nf $C2' --scan
case_ "L2 commit unpushed (tracking refs)"                      2  3 'base "$F" $C2' --scan
case_ "L2 commit unpushed (ls-remote)"                          2  3 'base "$F" $C2; REMOTE_CHECK=ls-remote' --scan
case_ "L2 tree not renamed (no python/flliper)"                 2  3 'base "$F" $C0' --scan
case_ "L2 commit unknown in repo"                               2  3 'X=$(printf "e%.0s" {1..40}); base "$F" $X' --scan
# ---- lock 3 -------------------------------------------------------------------------------------------------------------
case_ "L3 27B verdict names only the tag (arm as of today)"     3  3 'rm "$F/v27/facts_r1.kv"' --scan
case_ "L3 27B verdict bound to another image id"                3  3 'echo "image_id=$OTHER" > "$F/v27/facts_r1.kv"' --scan
case_ "L3 27B newest bound verdict FAIL"                        3  3 'sed -i "s/Overall: \*\*PASS\*\*/Overall: **FAIL**/" "$F/v27/verdict_r1.md"' --scan
case_ "L3 27B verdict OFFEN"                                    3  3 'sed -i "s/Overall: \*\*PASS\*\*/Overall: **OFFEN**/" "$F/v27/verdict_r1.md"' --scan
case_ "L3 27B verdict for another tree"                         3  3 'sed -i "s/27B tree \`[0-9a-f]*\`/27B tree \`${C0:0:10}\`/" "$F/v27/verdict_r1.md"' --scan
case_ "L3 27B verdict is not INT8 (NVFP4 heading)"              3  3 'sed -i "1s/INT8/NVFP4/" "$F/v27/verdict_r1.md"' --scan
case_ "L3 NF verdict not given"                                 3  3 'NFV=""' --scan
case_ "L3 NF verdict is NVFP4, not INT4"                        3  3 'sed -i "1s/INT4/NVFP4/" "$F/nf/verdict_nf.md"' --scan
case_ "L3 NF verdict without image id"                          3  3 'sed -i "s/Image id sha256:[0-9a-f]*/Image flliper-cand:test/" "$F/nf/verdict_nf.md"' --scan
# ---- lock 4 -------------------------------------------------------------------------------------------------------------
case_ "L4 Go file missing"                                      4  3 'rm "$F/go/$HEX.go"' --scan
case_ "L4 Go file without go: line"                             4  3 'sed -i "/^go:/d" "$F/go/$HEX.go"; touch -d "2026-09-26 19:00" "$F/go/$HEX.go"' --scan
case_ "L4 Go file for another digest"                           4  3 'sed -i "s/$ID/$OTHER/" "$F/go/$HEX.go"; touch -d "2026-09-26 19:00" "$F/go/$HEX.go"' --scan
case_ "L4 Go file older than the verdicts"                      4  3 'touch -d "2026-09-26 17:00" "$F/go/$HEX.go"' --scan
case_ "L4 Go file is a symlink"                                 4  3 'mv "$F/go/$HEX.go" "$F/go.real"; ln -s "$F/go.real" "$F/go/$HEX.go"' --scan
# ---- lock 5 -------------------------------------------------------------------------------------------------------------
case_ "L5 plan without --scan"                                  5  3 ':'
case_ "L5 token in a file under /opt"                           5  3 'echo "tok = \"$TOK_GH\"" >> "$F/img/opt/flliper/app.py"' --scan
case_ "L5 private key under /etc"                               5  3 'printf -- "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n" > "$F/img/etc/k"' --scan
case_ "L5 credential file /root/.git-credentials"               5  3 'echo "x" > "$F/img/root/.git-credentials"' --scan
case_ "L5 token in image env"                                   5  3 'echo "HF_TOKEN=$TOK_HF" >> "$F/env.txt"' --scan
case_ "L5 token in build history (ARG)"                         5  3 'echo "RUN |1 GITHUB_TOKEN=$TOK_GH /bin/sh -c true" >> "$F/history.txt"' --scan
case_ "L5 container scan cut (no SCAN-END)"                     5  3 'touch "$F/run_cut"' --scan
case_ "L5 docker run failed"                                    5  3 'touch "$F/run_fail"' --scan
case_ "L5 docker history failed"                                5  3 'touch "$F/history_fail"' --scan
case_ "L5 excluded dir hides a hit (exclusion is printed)"      -  0 'mkdir -p "$F/img/opt/fixtures"; echo "$TOK_GH" > "$F/img/opt/fixtures/t"; export SCAN_EXCLUDE_DIRS=fixtures' --scan
unset SCAN_EXCLUDE_DIRS
# ---- lock 6 -------------------------------------------------------------------------------------------------------------
case_ "L6 registry not given (B4 default)"                      6  3 'REGISTRY_ARG=()' --scan
case_ "L6 registry is htsglang"                                 6  3 'REGISTRY_ARG=(--registry ghcr.io/efschu/htsglang)' --scan
case_ "L6 registry uppercase"                                   6  3 'REGISTRY_ARG=(--registry ghcr.io/Efschu/flliper)' --scan
case_ "L6 version is a probe"                                   6  3 'setlabel "$F" org.opencontainers.image.version 0.0.0-probe1' --scan
case_ "L6 version uppercase"                                    6  3 'setlabel "$F" org.opencontainers.image.version 0.1.0-RC1' --scan
case_ "L6 cuda label cu129"                                     6  3 'setlabel "$F" io.github.efschu.flliper.cuda cu129' --scan
case_ "L6 release tag already in the registry"                  6  3 'echo "ghcr.io/efschu/flliper:0.1.0-rc1-cu130" > "$F/registry"' --scan
case_ "L6 local version tag points to another image"            6  3 'echo "flliper:0.1.0-rc1-cu130 $OTHER" >> "$F/tags"' --scan
# ---- publish ------------------------------------------------------------------------------------------------------------
case_ "PUBLISH refused with one red lock (Go missing)"          4  3 'rm "$F/go/$HEX.go"' --publish
case_ "PUBLISH refused, registry default"                       6  3 'REGISTRY_ARG=()' --publish
S10=${C1:0:10}
EXPECT_PUSH=1 WANT_ACT="TAG flliper:0.1.0-rc1-cu130 TAG flliper:cu130-$S10 TAG ghcr.io/efschu/flliper:cu130-$S10 PUSH ghcr.io/efschu/flliper:cu130-$S10 TAG ghcr.io/efschu/flliper:0.1.0-rc1-cu130 PUSH ghcr.io/efschu/flliper:0.1.0-rc1-cu130"
case_ "PUBLISH all green: tag local + registry, push sha then version" - 0 ':' --publish

echo "== $PASSC passed, $FAILC failed"
[ "$FAILC" = 0 ] || { printf '   failed: %s\n' "${FAILED[@]}"; exit 1; }
exit 0
