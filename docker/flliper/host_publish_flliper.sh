#!/bin/bash
# STAGED (item 600): host_publish_flliper.sh.new_600, derived from the live file sha256 efe37383d09e7e08b140fcb988707188784a8744e55e2832b86bf84990fab0b4 -- install with install_600.sh
# host_publish_flliper.sh -- publish gate for the flat fLLiper release image (blocker B5, FLAT_IMAGE_PLAN.md §6).
# PB for the 27B operator, 2026-09-26. NEW file; host_publish.sh (htsglang, single-SHA lines) stays as it is.
#
#   IMAGE=<local tag or sha256:id> NF_VERDICT=<NF-INT4 verdict .md> \
#     bash host_publish_flliper.sh [--plan] [--scan] [--publish] [--registry <repo>]
#
#   --plan     (default) check all six locks, print green/red with the reason per lock. Changes nothing: no tag, no push,
#              no login, no file written. The secret scan (lock 5) runs only with --scan (it starts one throw-away
#              container that reads /opt /etc /root of the image); without --scan lock 5 is red "not scanned".
#   --publish  same checks, secret scan always; REFUSES (rc 3) unless every lock is green. Then:
#              docker tag <id> flliper:<version>-cu130 / flliper:cu130-<sha10> (local, only if missing) and
#              docker tag + docker push <registry>:cu130-<sha10>, <registry>:<version>-cu130. It never logs in: without a
#              prior `docker login` by the user the push fails. Run on the Proxmox host (docker lives there).
#
# LOCKS (all must be green for --publish):
#   1 labels     every OCI + io.github.efschu.flliper.* label of Dockerfile.flliper present and non-empty (F0-G: incl. .revision = the OCI revision); label
#                io.github.efschu.flliper.placeholders empty or "none"; no value carries a placeholder
#                (UNDECIDED / PLACEHOLDER / __FLLIPER_); title fLLiper, slots "27b nf", build flat, source https://,
#                lock/kernel-wheel sha256 = 64 hex; no leftover htsglang.* label.
#   2 revision   org.opencontainers.image.revision is a 40-hex SHA, equal to revision.27b; revision.nf equal to it (ONE tree) or
#                another 40-hex SHA (item 600, Duo: two renamed trees, each checked below; immutable tag cu130-<27b10>-<nf10>);
#                the commit exists in $REPO; it is on the remote $REMOTE -- REMOTE_CHECK=tracking (default): a
#                remote-tracking ref refs/remotes/$REMOTE/* contains it (fetch proof, FETCH_HEAD time printed);
#                REMOTE_CHECK=ls-remote: a tip listed by `git ls-remote $REMOTE` is the SHA or a local descendant of it;
#                the tree is renamed (python/flliper is a tree at that SHA).
#   3 acceptance PASS verdict for 27B-INT8 (newest verdict_*.md in $VERDICT_27B_DIR bound to this image, from
#                acc_cu130_final_27b.sh; or VERDICT_27B=<file>) AND for NF-INT4 (NF_VERDICT=<file>, from the NF seat),
#                both bound to EXACTLY this image id: the verdict file, or its sibling facts_<RUNID>.kv
#                (image_id= / image_digest=), must carry the full sha256:<64 hex> image id. A tag alone is no binding.
#                Heading: 27B file names "27B" and "INT8", NF file names "NF" and "INT4" (first heading only;
#                user order 26.09.: first docker acceptance only 27B-INT8 + NF-INT4). Every "Overall:"/"Gesamt:" verdict
#                in the file must be PASS. A "27B tree `<sha>`" / "tree `<sha>`" mention must match the revision.
#   4 user Go    $GO_DIR/<64-hex image id>.go exists (regular file, not a symlink), names the full image id and has a
#                non-empty "go:" line (the user's verbatim Go, entered by the operator); not older than the verdicts.
#                THIS SCRIPT NEVER CREATES, TOUCHES OR EDITS THAT FILE (memory docker-release-entscheid-0924).
#   5 secrets    image env, labels and build history, plus a read-only container scan of $SCAN_PATHS
#                (docker run --rm --network none --read-only), against token patterns (GitHub, Anthropic, OpenRouter,
#                OpenAI-style, HuggingFace, AWS, Slack, GitLab, npm, Google, private keys, URL credentials) and
#                credential file names. Only COUNTS leave the container; a match is never printed, not even a file name.
#   6 tags       version label is a lowercase tag part without probe/sglang/weg2/htsglang/27b-nf; cuda label cu130;
#                tags flliper:<version>-cu130 + flliper:cu130-<sha10>; registry given EXPLICITLY (the default
#                ghcr.io/efschu/flliper is a B4 placeholder), lowercase, ends in /flliper, no htsglang; neither local tag
#                points to another image; neither registry tag exists yet (docker manifest inspect; without a login a
#                private repo cannot tell "missing" from "forbidden" -- the push itself would then still fail, nothing is
#                overwritten).
#
# KNOBS (env): IMAGE NF_VERDICT VERDICT_27B VERDICT_27B_DIR REGISTRY REPO REMOTE(origin) REMOTE_CHECK(tracking|ls-remote)
#   GO_DIR SCAN_PATHS("/opt /etc /root") SCAN_EXCLUDE_DIRS(grep --exclude-dir globs, printed; default none)
#   SCAN_ALLOW (ERE of public placeholder tokens that do not count, printed) S_ROOT (CT999 root as the host sees it;
#   auto: /spinning/subvol-999-disk-0 if present, else "")
# Exit: 0 = all green (--plan) / pushed (--publish); 3 = red lock(s), nothing published; 2 = usage; 1 = push failed.
# Self-test (fake docker, no real docker): bash host_publish_flliper_selftest.sh
set -uo pipefail

MODE=plan; SCAN=0; REG_EXPLICIT=0
[ -n "${REGISTRY:-}" ] && REG_EXPLICIT=1
while [ $# -gt 0 ]; do
  case "$1" in
    --plan) MODE=plan; shift ;;
    --publish) MODE=publish; shift ;;
    --scan) SCAN=1; shift ;;
    --image) IMAGE=$2; shift 2 ;;
    --nf-verdict) NF_VERDICT=$2; shift 2 ;;
    --verdict-27b) VERDICT_27B=$2; shift 2 ;;
    --registry) REGISTRY=$2; REG_EXPLICIT=1; shift 2 ;;
    -h|--help) sed -n '2,52p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help); nothing was done" >&2; exit 2 ;;
  esac
done
[ "$MODE" = publish ] && SCAN=1
[ -n "${IMAGE:-}" ] || { echo "IMAGE (local tag or sha256:id) missing; nothing was done" >&2; exit 2; }

if [ -n "${S_ROOT+x}" ]; then S=$S_ROOT
elif [ -d /spinning/subvol-999-disk-0/spinning/gpu-arb/docker ]; then S=/spinning/subvol-999-disk-0
else S=""; fi
CU=cu130
REGISTRY=${REGISTRY:-ghcr.io/efschu/flliper}
REPO=${REPO:-$S/spinning/htsglang}
REMOTE=${REMOTE:-origin}
REMOTE_CHECK=${REMOTE_CHECK:-tracking}
VERDICT_27B=${VERDICT_27B:-}
VERDICT_27B_DIR=${VERDICT_27B_DIR:-$S/spinning/docker-acceptance/27b/final27b}
NF_VERDICT=${NF_VERDICT:-}
GO_DIR=${GO_DIR:-$S/spinning/gpu-arb/docker/publish_go}
SCAN_PATHS=${SCAN_PATHS:-/opt /etc /root}
SCAN_EXCLUDE_DIRS=${SCAN_EXCLUDE_DIRS:-}
ALLOW_DEFAULT='EXAMPLE|X{8,}|x{8,}|0{20,}|[.]{3}|<[A-Za-z_-]+>|[$][{]?[A-Za-z_]+'
SCAN_ALLOW=${SCAN_ALLOW:-$ALLOW_DEFAULT}
G=(git --no-optional-locks -c safe.directory='*' -C "$REPO")

say(){ printf '%s\n' "$*"; }
declare -A RED WHY
red(){ RED[$1]=1; WHY[$1]="${WHY[$1]:+${WHY[$1]}; }$2"; say "   red: $2"; }
ok(){ say "   ok: $*"; }
sec(){ printf '\n== %s\n' "$*"; }
mtime(){ stat -c %Y "$1" 2>/dev/null || echo 0; }
utc(){ date -u -d "@$1" +%FT%TZ 2>/dev/null || echo "?"; }
nocred(){ sed -E 's#(://)[^/@]*@#\1***@#'; }   # never print URL credentials

say "host_publish_flliper.sh --$MODE  $(date -u +%FT%TZ)  $([ "$MODE" = plan ] && echo '(read-only: no tag, no push, no login, no file written)' || echo '(publishes ONLY if all six locks are green)')"

# ---- image ------------------------------------------------------------------------------------------------------------
ID=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null)
case "$ID" in sha256:[0-9a-f]*) ;; *) say "   image $IMAGE not found locally (docker image inspect) -- all locks red"; say "RESULT: 6/6 red -- nothing published"; exit 3 ;; esac
HEX=${ID#sha256:}
case "$HEX" in *[!0-9a-f]*) HEX="" ;; esac
[ ${#HEX} = 64 ] || { say "   image id '$ID' is not sha256:<64 hex> -- all locks red"; exit 3; }
declare -A L
while IFS= read -r line; do
  [ -n "$line" ] || continue
  L["${line%%=*}"]=${line#*=}
done < <(docker image inspect -f '{{range $k, $v := .Config.Labels}}{{$k}}={{$v}}{{println}}{{end}}' "$ID" 2>/dev/null)
say "   image $IMAGE = $ID (${#L[@]} labels)"

# ---- lock 1: labels ---------------------------------------------------------------------------------------------------
sec "1. Labels complete, no placeholders"
OCI=org.opencontainers.image; P=io.github.efschu.flliper
REQ=($OCI.title $OCI.version $OCI.revision $OCI.source $OCI.licenses $OCI.description $OCI.base.name
     $P.slots $P.revision $P.revision.27b $P.revision.nf $P.push_state $P.cuda $P.kernel_wheel.sha256 $P.kernel_archs
     $P.flashinfer $P.cute_dsl $P.nccl $P.driver.expected $P.lock.sha256 $P.build)
miss=()
for k in "${REQ[@]}"; do [ -n "${L[$k]:-}" ] || miss+=("$k"); done
[ ${#miss[@]} = 0 ] && ok "${#REQ[@]} required labels present" || red 1 "missing/empty: ${miss[*]}"
if [ -z "${L[$P.placeholders]+x}" ]; then red 1 "label $P.placeholders absent (a flat fLLiper image always carries it)"
else case "${L[$P.placeholders]}" in ""|none) ok "$P.placeholders='${L[$P.placeholders]}' (no open B4 value)" ;;
     *) red 1 "$P.placeholders='${L[$P.placeholders]}' -- open B4 value(s), rebuild with --source/--release" ;; esac; fi
phv=()
for k in "${!L[@]}"; do
  case "$k" in "$OCI".*|"$P".*) ;; *) continue ;; esac
  printf '%s' "${L[$k]}" | grep -qiE 'UNDECIDED|PLACEHOLDER|__FLLIPER_' && phv+=("$k")
done
[ ${#phv[@]} = 0 ] && ok "no placeholder value in any label" || red 1 "placeholder value in: ${phv[*]}"
[[ "${L[$OCI.revision]:-}" =~ ^[0-9a-f]{40}$ ]] && { [ "${L[$P.revision]:-}" = "${L[$OCI.revision]:-}" ] || red 1 "label $P.revision '${L[$P.revision]:-}' != $OCI.revision '${L[$OCI.revision]:-}' (F0-G: both name the 27B tree)"; }   # a malformed OCI revision is lock 2's finding
[ "${L[$OCI.title]:-}" = fLLiper ] || red 1 "title '${L[$OCI.title]:-}' != fLLiper"
[ "${L[$P.slots]:-}" = "27b nf" ] || red 1 "slots '${L[$P.slots]:-}' != '27b nf'"
[ "${L[$P.build]:-}" = flat ] || red 1 "build '${L[$P.build]:-}' != flat (release image must be the flat build)"
case "${L[$OCI.source]:-}" in https://?*) ;; *) red 1 "source '${L[$OCI.source]:-}' is not an https URL" ;; esac
for k in $P.lock.sha256 $P.kernel_wheel.sha256; do
  [[ "${L[$k]:-}" =~ ^[0-9a-f]{64}$ ]] || red 1 "$k '${L[$k]:-}' is not 64 hex"
done
old=$(for k in "${!L[@]}"; do case "$k" in htsglang.*) printf '%s ' "$k" ;; esac; done)
[ -z "$old" ] && ok "no htsglang.* label" || red 1 "leftover htsglang.* labels: $old"
[ -z "${RED[1]:-}" ] && WHY[1]="labels complete, placeholders '${L[$P.placeholders]}'"

# ---- lock 2: revision -------------------------------------------------------------------------------------------------
sec "2. Revision: 40-hex SHA, one tree, on the remote, renamed"
REV=${L[$OCI.revision]:-}
if ! [[ "$REV" =~ ^[0-9a-f]{40}$ ]]; then red 2 "revision label '$REV' is not a plain 40-hex SHA"; REV=""
else
  ok "revision $REV"
  # Item 600: the image may carry TWO renamed trees (Duo: revision.27b = revision, revision.nf = the NF tree). Each must be renamed and on the remote.
  DUO=0; REVNF=${L[$P.revision.nf]:-}
  if [ "${L[$P.revision.27b]:-}" != "$REV" ]; then red 2 "revision.27b '${L[$P.revision.27b]:-}' != revision '$REV' (the OCI revision is the 27B tree)"
  elif [ "$REVNF" = "$REV" ]; then ok "revision.27b = revision.nf = revision (one tree)"
  elif [[ "$REVNF" =~ ^[0-9a-f]{40}$ ]]; then DUO=1; ok "DUO: revision.27b = revision ${REV:0:10}, revision.nf = $REVNF (two renamed trees)"
  else red 2 "slot revisions: nf '$REVNF' is neither the 27b revision nor a plain 40-hex SHA"; fi
  RVS="$REV"; [ "$DUO" = 1 ] && RVS="$REV $REVNF"
  for RV in $RVS; do
    if ! "${G[@]}" cat-file -e "$RV^{commit}" 2>/dev/null; then red 2 "commit ${RV:0:10} unknown in $REPO (fetch first)"
    else
      if [ "$("${G[@]}" cat-file -t "$RV:python/flliper" 2>/dev/null)" = tree ]; then ok "renamed tree: python/flliper present at ${RV:0:10}"
      else red 2 "tree ${RV:0:10} is not renamed (python/flliper missing, B1)"; fi
      RURL=$("${G[@]}" remote get-url "$REMOTE" 2>/dev/null | nocred)
      [ -n "$RURL" ] || red 2 "remote '$REMOTE' unknown in $REPO"
      if [ -n "$RURL" ] && [ "$REMOTE_CHECK" = ls-remote ]; then
        hit=""
        while read -r tip ref; do
          [ -n "$tip" ] || continue
          if [ "$tip" = "$RV" ] || { "${G[@]}" cat-file -e "$tip^{commit}" 2>/dev/null && "${G[@]}" merge-base --is-ancestor "$RV" "$tip" 2>/dev/null; }; then hit="$ref"; break; fi
        done < <(GIT_TERMINAL_PROMPT=0 "${G[@]}" ls-remote "$REMOTE" 2>/dev/null)
        [ -n "$hit" ] && ok "on the remote $REMOTE ($RURL): $hit contains ${RV:0:10} (ls-remote $(date -u +%FT%TZ))" \
                      || red 2 "ls-remote $REMOTE ($RURL): no listed tip is ${RV:0:10} or a known descendant (unpushed, or ls-remote failed -- its stderr is not printed)"
      elif [ -n "$RURL" ] && [ "$REMOTE_CHECK" = tracking ]; then
        refs=$("${G[@]}" for-each-ref --format='%(refname:short)' --contains "$RV" "refs/remotes/$REMOTE/" 2>/dev/null | grep -v '/HEAD$' | head -3 | paste -sd, -)
        GD=$("${G[@]}" rev-parse --absolute-git-dir 2>/dev/null)
        FH=$(mtime "$GD/FETCH_HEAD")
        [ -n "$refs" ] && ok "on the remote $REMOTE ($RURL): $refs (fetch proof, FETCH_HEAD $(utc "$FH"))" \
                       || red 2 "no refs/remotes/$REMOTE/* contains ${RV:0:10} (UNPUSHED, or no fetch since the push; last FETCH_HEAD $(utc "$FH"))"
      elif [ -n "$RURL" ]; then red 2 "REMOTE_CHECK='$REMOTE_CHECK' unknown (tracking|ls-remote)"; fi
    fi
  done
  say "   info: build-time push_state label: '${L[$P.push_state]:-}' (git decides, not the label)"
fi
[ -z "${RED[2]:-}" ] && WHY[2]="${REV:0:10} on $REMOTE, renamed"

# ---- lock 3: acceptance verdicts --------------------------------------------------------------------------------------
sec "3. Acceptance PASS for 27B-INT8 and NF-INT4 on image ${HEX:0:12}"
# bound <verdict>: 0 if the verdict file or its facts_<RUNID>.kv carries the full image id
bound(){ local f=$1 d r fx
  grep -qF "$ID" "$f" 2>/dev/null && { echo "id in verdict"; return 0; }
  d=$(dirname "$f"); r=$(basename "$f" .md); r=${r#verdict_}; fx=$d/facts_$r.kv
  [ -f "$fx" ] && grep -qxE "(image_id|image_digest)=$ID" "$fx" && { echo "id in $(basename "$fx")"; return 0; }
  return 1; }
# judge <n> <label> <file> <heading-re-1> <heading-re-2> <tree-re>
judge(){ local n=$1 lab=$2 f=$3 h1=$4 h2=$5 tre=$6 rv=${7:-$REV} head ov bad t how w0=${WHY[$1]:-}
  [ -f "$f" ] && [ ! -L "$f" ] || { red "$n" "$lab verdict '$f' missing (or a symlink)"; return 1; }
  how=$(bound "$f") || { red "$n" "$lab verdict $(basename "$f") is not bound to $ID (neither the file nor facts_<RUNID>.kv names the image id)"; return 1; }
  head=$(grep -m1 -E '^#' "$f")
  { printf '%s' "$head" | grep -qiE "$h1" && printf '%s' "$head" | grep -qiE "$h2"; } || red "$n" "$lab verdict heading '${head:0:80}' does not name $h1 + $h2"
  ov=$(grep -oE '(Overall|Gesamt|GESAMT): *[*]{0,2}[A-Z]+' "$f" | grep -oE '[A-Z]+$' | sort -u | paste -sd, -)
  if [ -z "$ov" ]; then red "$n" "$lab verdict $(basename "$f") has no 'Overall: **...**' line"
  elif [ "$ov" != PASS ]; then red "$n" "$lab verdict $(basename "$f") overall '$ov' (must be PASS only)"; fi
  bad=""
  for t in $(grep -oE "$tre"'`[0-9a-f]{7,40}`' "$f" | grep -oE '[0-9a-f]{7,40}' | sort -u); do
    [ -n "$rv" ] && [ "${rv#"$t"}" != "$rv" ] || bad="$bad $t"
  done
  [ -z "$bad" ] || red "$n" "$lab verdict names tree(s)$bad, not the image revision ${rv:0:10}${rv:+}$([ -n "$rv" ] || echo '(no valid revision label)')"
  [ "${WHY[$n]:-}" = "$w0" ] && ok "$lab: $(basename "$f") ($how, overall ${ov:-?}, $(utc "$(mtime "$f")"))"
  return 0
}
V27=$VERDICT_27B
if [ -z "$V27" ]; then
  nall=0
  while IFS= read -r f; do
    nall=$((nall + 1)); bound "$f" >/dev/null && { V27=$f; break; }
  done < <(ls -1t "$VERDICT_27B_DIR"/verdict_*.md 2>/dev/null)
  [ -n "$V27" ] || red 3 "27B-INT8: none of $nall verdict_*.md in $VERDICT_27B_DIR is bound to $ID (acc_cu130_final_27b.sh records only the image TAG today -- it needs a fact image_id=\$(docker image inspect -f '{{.Id}}' \$IMAGE), or the id in the verdict)"
fi
[ -n "$V27" ] && judge 3 27B-INT8 "$V27" '27b' 'int8' '27B tree '
if [ -z "$NF_VERDICT" ]; then red 3 "NF-INT4: NF_VERDICT (verdict file from the NF seat) not given"
else judge 3 NF-INT4 "$NF_VERDICT" '(^|[^a-z])nf([^a-z]|$)' 'int4' 'tree ' "${REVNF:-$REV}"; fi
[ -z "${RED[3]:-}" ] && WHY[3]="27B-INT8 $(basename "$V27") + NF-INT4 $(basename "$NF_VERDICT") PASS on ${HEX:0:12}"

# ---- lock 4: user Go --------------------------------------------------------------------------------------------------
sec "4. User Go for exactly this image (file written by the operator, never by this script)"
GOF=$GO_DIR/$HEX.go
if [ -L "$GOF" ]; then red 4 "$GOF is a symlink"
elif [ ! -f "$GOF" ]; then red 4 "no Go file $GOF (the operator creates it after the user's verbatim Go for this digest)"
else
  grep -qF "$ID" "$GOF" || red 4 "$(basename "$GOF") does not name the full image id $ID"
  gl=$(grep -m1 -E '^go:[[:space:]]*[^[:space:]]' "$GOF")
  [ -n "$gl" ] || red 4 "$(basename "$GOF") has no non-empty 'go:' line (user's verbatim Go)"
  gm=$(mtime "$GOF")
  for f in "$V27" "$NF_VERDICT"; do
    [ -n "$f" ] && [ -f "$f" ] && [ "$gm" -lt "$(mtime "$f")" ] && red 4 "Go file ($(utc "$gm")) is older than verdict $(basename "$f") ($(utc "$(mtime "$f")")) -- the Go is given per accepted state"
  done
  [ -n "$gl" ] && say "   Go: ${gl:0:160}"
fi
[ -z "${RED[4]:-}" ] && WHY[4]="$(basename "$GOF") $(utc "$(mtime "$GOF")")"

# ---- lock 5: secret scan ----------------------------------------------------------------------------------------------
sec "5. Secret scan (counts only; no match, no file name is ever printed)"
PATS=$(cat <<'EOF'
github	(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}
github-pat	github_pat_[A-Za-z0-9_]{50,}
anthropic	sk-ant-[A-Za-z0-9_-]{20,}
openrouter	sk-or-v1-[0-9a-f]{64}
openai-style	sk-(proj-)?[A-Za-z0-9]{40,}
huggingface	hf_[A-Za-z0-9]{34,}
aws-key	AKIA[0-9A-Z]{16}
slack	xox[abprs]-[A-Za-z0-9-]{10,}
gitlab	glpat-[A-Za-z0-9_-]{20,}
npm	npm_[A-Za-z0-9]{36}
google-api	AIza[0-9A-Za-z_-]{35}
private-key	-----BEGIN ([A-Z]+ )?PRIVATE KEY-----
url-credential	[a-z]+://[^/[:space:]:@"']+:[^/[:space:]@"']{8,}@
EOF
)
ENVPAT='^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|GITHUB_TOKEN|GH_TOKEN|[A-Z_]*API_KEY|[A-Z_]*PASSWORD|[A-Z_]*SECRET[A-Z_]*|[A-Z_]*AUTH_TOKEN)=[^$[:space:]]'
say "   patterns: $(printf '%s\n' "$PATS" | cut -f1 | paste -sd' ' -), credential file names, env assignments"
say "   not counted (public placeholders): SCAN_ALLOW='$SCAN_ALLOW'; excluded dirs: '${SCAN_EXCLUDE_DIRS:-none}'"
count_pats(){ local in n tot=0 out="" cls pat
  in=$(cat | grep -oE -f <(printf '%s\n' "$PATS" | cut -f2-) | grep -viE -e "$SCAN_ALLOW")
  while IFS=$'\t' read -r cls pat; do
    n=$(printf '%s\n' "$in" | grep -cE -e "$pat"); [ "${n:-0}" -gt 0 ] && out="$out $cls=$n"; tot=$((tot + ${n:-0}))
  done <<<"$PATS"
  echo "$tot${out}"; }
m=$(docker image inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$ID" 2>/dev/null)
c1=$(printf '%s\n' "$m" | count_pats); c1e=$(printf '%s\n' "$m" | grep -cE "$ENVPAT")
c2=$(docker image inspect -f '{{range $k, $v := .Config.Labels}}{{$v}}{{println}}{{end}}' "$ID" 2>/dev/null | count_pats)
hist=$(docker history --no-trunc --format '{{.CreatedBy}}' "$ID" 2>/dev/null) || hist="__history_failed__"
if [ "$hist" = __history_failed__ ]; then red 5 "docker history failed -- build history not scanned"; c3=0; c3e=0
else c3=$(printf '%s\n' "$hist" | count_pats); c3e=$(printf '%s\n' "$hist" | tr ' ' '\n' | grep -cE "$ENVPAT"); fi
for x in "env:${c1%% *}:${c1#* }" "env-assign:$c1e:" "labels:${c2%% *}:${c2#* }" "history:${c3%% *}:${c3#* }" "history-assign:$c3e:"; do
  IFS=: read -r where n det <<<"$x"; [ "$n" = "$det" ] && det=""
  [ "${n:-0}" = 0 ] && ok "$where: 0 hits" || red 5 "$where: $n hit(s)${det:+ ($det)}"
done
if [ "$SCAN" = 1 ]; then
  EXCL=""; for e in $SCAN_EXCLUDE_DIRS; do EXCL="$EXCL --exclude-dir=$e"; done
  # the pattern list goes in via env and becomes a file on the tmpfs (/tmp/p), so no ERE meets word splitting;
  # every match stays inside the container (/tmp/t on the tmpfs, gone with --rm); only "<class> <count>" lines leave it
  SCAN_SH='set -f
printf "%s\n" "$SCAN_PATTERNS" | cut -f2- > /tmp/p
grep -rIohsE $EXCL -f /tmp/p -- $SCAN_PATHS 2>/dev/null | grep -viE -e "$SCAN_ALLOW" > /tmp/t
printf "%s\n" "$SCAN_PATTERNS" | while IFS="	" read -r cls pat; do echo "$cls $(grep -cE -e "$pat" /tmp/t)"; done
echo "credfiles $(find $SCAN_PATHS -xdev -type f \( -name .git-credentials -o -name .netrc -o -name .pypirc -o -name id_rsa -o -name id_dsa -o -name id_ecdsa -o -name id_ed25519 -o -path "*/.docker/config.json" \) 2>/dev/null | wc -l)"
echo SCAN-END'
  t0=$(date +%s)
  res=$(docker run --rm --network none --read-only --tmpfs /tmp:rw,size=512m --user 0 --entrypoint /bin/sh \
        -e SCAN_PATTERNS="$PATS" -e SCAN_ALLOW="$SCAN_ALLOW" -e SCAN_PATHS="$SCAN_PATHS" -e EXCL="$EXCL" \
        "$ID" -c "$SCAN_SH" 2>/dev/null)
  if ! printf '%s\n' "$res" | grep -qx SCAN-END; then red 5 "container scan incomplete (no SCAN-END; docker run failed or was cut) -- not a clean result"
  else
    tot=0; det=""
    while read -r cls n; do
      [ "$cls" = SCAN-END ] && continue
      [[ "$n" =~ ^[0-9]+$ ]] || { red 5 "scan line for '$cls' unreadable"; continue; }
      [ "$n" -gt 0 ] && det="$det $cls=$n"; tot=$((tot + n))
    done <<<"$res"
    [ "$tot" = 0 ] && ok "container scan of $SCAN_PATHS: 0 hits ($(( $(date +%s) - t0 )) s)" \
                   || red 5 "container scan of $SCAN_PATHS: $tot hit(s) ($det ) -- inspect inside the image yourself; never paste a match"
  fi
else
  red 5 "container scan not run (--plan without --scan); --publish always scans"
fi
[ -z "${RED[5]:-}" ] && WHY[5]="env/labels/history/container 0 hits"

# ---- lock 6: tags -----------------------------------------------------------------------------------------------------
sec "6. Tag schema flliper:<version>-$CU + flliper:$CU-<sha10>, registry"
VER=${L[$OCI.version]:-}
if ! [[ "$VER" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then red 6 "version '$VER' is not a lowercase tag part [a-z0-9._-]"
elif printf '%s' "$VER" | grep -qE 'probe|sglang|weg2|htsglang|27b-nf'; then red 6 "version '$VER' carries a forbidden word (probe/sglang/weg2/htsglang/27b-nf)"
else ok "version $VER"; fi
[ "${L[$P.cuda]:-}" = "$CU" ] || red 6 "cuda label '${L[$P.cuda]:-}' != $CU"
SHA10=${REV:0:10}; [ -n "$SHA10" ] || { SHA10=unknown; red 6 "no valid revision -> no cu130-<sha10> tag"; }
[ "${DUO:-0}" = 1 ] && SHA10="${REV:0:10}-${REVNF:0:10}"   # item 600: the immutable tag names both trees (27B-NF)
T1="flliper:$VER-$CU"; T2="flliper:$CU-$SHA10"; R1="$REGISTRY:$VER-$CU"; R2="$REGISTRY:$CU-$SHA10"
[ "$REG_EXPLICIT" = 1 ] || red 6 "registry not given (default $REGISTRY is a B4 placeholder) -- pass --registry after the user's decision"
if ! [[ "$REGISTRY" =~ ^[a-z0-9.-]+(:[0-9]+)?(/[a-z0-9._-]+)+$ ]]; then red 6 "registry '$REGISTRY' is not a lowercase repository reference"
elif [ "${REGISTRY##*/}" != flliper ]; then red 6 "registry '$REGISTRY' does not end in /flliper"
elif printf '%s' "$REGISTRY" | grep -q htsglang; then red 6 "registry '$REGISTRY' is an htsglang repository (old tags stay untouched)"
else ok "registry $REGISTRY"; fi
for t in "$T1" "$T2"; do
  cur=$(docker image inspect -f '{{.Id}}' "$t" 2>/dev/null)
  if [ -z "$cur" ]; then ok "local $t free (will be tagged)"
  elif [ "$cur" = "$ID" ]; then ok "local $t already = this image"
  else red 6 "local $t points to another image ${cur:7:12} -- never re-pointed (F14)"; fi
done
for t in "$R2" "$R1"; do
  if docker manifest inspect "$t" >/dev/null 2>&1; then red 6 "$t exists in the registry -- never overwritten, choose a new version"
  else ok "$t not in the registry (or not readable without login)"; fi
done
say "   tags: $T1  $T2  ->  $R2  $R1"
[ -z "${RED[6]:-}" ] && WHY[6]="$R1 + $R2"

# ---- verdict ----------------------------------------------------------------------------------------------------------
NAMES=("" labels revision acceptance user-go secrets tags)
sec "Locks"
nred=0
for n in 1 2 3 4 5 6; do
  if [ -n "${RED[$n]:-}" ]; then nred=$((nred + 1)); say "   $n ${NAMES[$n]}: RED   -- ${WHY[$n]}"
  else say "   $n ${NAMES[$n]}: green -- ${WHY[$n]}"; fi
done
if [ "$nred" -gt 0 ]; then
  say "RESULT: $nred/6 red -- $([ "$MODE" = publish ] && echo 'PUBLISH REFUSED, nothing tagged or pushed' || echo 'nothing done (plan)')"
  exit 3
fi
if [ "$MODE" = plan ]; then
  say "RESULT: 6/6 green (plan) -- would run:"
  say "   docker tag $ID $T1; docker tag $ID $T2; docker tag $ID $R2; docker push $R2; docker tag $ID $R1; docker push $R1"
  exit 0
fi

# ---- publish (all six green) ------------------------------------------------------------------------------------------
sec "Publish"
now=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null)
[ "$now" = "$ID" ] || { say "REFUSED: $IMAGE now resolves to ${now:-nothing}, not $ID (changed during the checks)"; exit 3; }
for t in "$T1" "$T2"; do [ "$(docker image inspect -f '{{.Id}}' "$t" 2>/dev/null)" = "$ID" ] || docker tag "$ID" "$t" || { say "FAILED: docker tag $t"; exit 1; }; done
for t in "$R2" "$R1"; do
  docker tag "$ID" "$t" || { say "FAILED: docker tag $t"; exit 1; }
  say "   \$ docker push $t"
  docker push "$t" || { say "FAILED: docker push $t (not logged in? the user logs in himself; nothing is retried)"; exit 1; }
done
say "RESULT: published $R2 and $R1 = $ID ($(docker image inspect -f '{{range .RepoDigests}}{{.}} {{end}}' "$ID" 2>/dev/null))"
exit 0
