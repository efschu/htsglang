#!/usr/bin/env bash
# htsglang Release-Entrypoint (Upgrade des August-Containers) -- ENTWURF (27B-Sitz R, 24.09.2026).
# NICHT GEBAUT, NICHT GELAUFEN. Nur `bash -n` geprueft.
#
# DISPATCHER. MODE=server|planner (oder MODE ungesetzt) -> unveraendert das August-Entrypoint
# /usr/local/bin/htsglang-entrypoint.sh (docker/htsglang-entrypoint.sh, env-getriebenes
# launch_server bzw. Planner). Bestehende `docker run`-Zeilen des veroeffentlichten Containers
# laufen also weiter wie bisher. NEU: MODE=weg2 -> der weg2-P/D-Flip-Launcher mit Profil.
#
# weg2-Ablauf:
#  1. Profil laden (FLLIPER_PROFILE oder HTSGLANG_PROFILE: 27b, 27b-fp8, 27b-nvfp4, 27b-gguf | nf, nf-nvfp4, nf-gguf);
#     Profil-Linie == Image-Linie; Platzhalter-Profile verweigern mit Verweis auf den Eigentuemer.
#  2. Modellformat am gemounteten Modell erkennen und gegen PROFILE_FORMAT pruefen.
#  3. Umgebung: FLASHINFER_CUDA_ARCH_LIST und TORCH_CUDA_ARCH_LIST weg (JIT-Vorbau-Namen),
#     expandable_segments verweigert (launcher.py:5515).
#  4. Preflight: GPU-Inventar, /dev/shm, Speicher, Baum-Identitaet, Modelle, Zustands-Dirs
#     (Stufe B: SGLANG_WEG2_*), ARB-Saat, tmpfs-Store (NF), BAR1-Kette.
#  5. Transport (Operator-Entscheid F7, 24.09.): bar1 (DEFAULT) | nccl (nur ausdruecklich).
#     Kein stilles Umschalten und kein `auto`: bar1 ohne vollstaendige Kette = Verweigerung mit
#     Grund; nccl = barlink KOMPLETT aus (Launcher --transport nccl streicht die barlink-Flags
#     beider Gruppen, hier zusaetzlich SGLANG_BARLINK/_TRANSPORT/_PP_TRANSPORT aus der Umgebung).
#     Beide Transporte muessen in der Host-Abnahme booten (host_acceptance.sh serve bar1|nccl).
#  Instrumente (F8): HTSGLANG_INSTRUMENTS=0 ist der Release-Default; =1 schaltet die
#     Mess-Env des Profils zu (Paritaet mit den Rig-Referenzboots, Abnahme).
#  6. Launcher starten; er kehrt nach "LAUNCHED" mit rc 0 zurueck (launcher.py:14068-14069) ->
#     dieser Prozess beaufsichtigt die Front und baut ueber `--teardown <state.json>` ab.
#
# weg2-Untermodi (erstes Argument oder HTSGLANG_MODE): serve (Default) | dryrun | preflight |
#   selfcheck | version | bar1probe [karten slot_mib].  Weitere Argumente gehen bei serve/dryrun unveraendert
#   an den Launcher. bar1probe = Preflight + benchmark/bar1_graph_check.py in DERSELBEN Umgebung wie serve
#   (Arch-Liste weg, Kette geprueft; Default 0,1,2 29700) -- Abnahme T0b, und der BAR1-Test fuer Nutzer-Hosts.

set -Eeuo pipefail
umask 022

# >>> ENTRYPOINT-NAMES (fLLiper Schritt 5, EP 26.09.; nur Definitionen -- selftest: docker/release/selftest_names_ep.sh)
# README <-> Image. Der README (fLLiper v0.1) nennt FLLIPER_*, /var/lib/flliper/* und /root/.cache/flliper; host_acceptance.sh
# und alle Arme fahren HTSGLANG_*, /var/lib/htsglang/* und /root/.cache/sglang. Beide Namen gelten:
#  * Produkt-Env: FLLIPER_<X> wird auf HTSGLANG_<X> abgebildet (der Rest dieses Skripts, die Profile und host_acceptance
#    lesen weiter HTSGLANG_<X>), danach wird FLLIPER_<X> ENTFERNT: FLLIPER_<X> ist zugleich die umbenannte Schreibweise
#    der Laufzeit-Variable SGLANG_<X> (name_compat.ENV_PREFIX_PAIRS). Beispiel FLLIPER_P_CHUNK_POLICY = Produktschalter
#    UND umbenanntes SGLANG_P_CHUNK_POLICY (weg2/p_chunk_policy.py POLICY_ENV); gesetzt gelassen erreichte es jeden Rang als
#    Laufzeitwert, FLLIPER_<X> verhielte sich dann nicht byte-gleich zu HTSGLANG_<X>.
#    Ausnahme (RENAME_PLAN 3.4): HTSGLANG_TAG <-> FLLIPER_BOOT_TAG (FLLIPER_TAG kollidierte mit SGLANG_TAG).
#  * Zustands-Pfade: FLLIPER_PDFLIP_<X> und SGLANG_WEG2_<X> (EVIDENCE_DIR, GPU_ARB, STORE_ROOT); ohne Variable entscheidet
#    das Volume (/var/lib/flliper/{evidence,arb,hicache} oder /var/lib/htsglang/{evidence,arb,hicache-weg2}). Beide
#    Schreibweisen werden mit DEMSELBEN Wert exportiert (alter Baum liest SGLANG_WEG2_*, umbenannter FLLIPER_PDFLIP_*).
#  * Cache: /root/.cache/flliper oder /root/.cache/sglang als Volume; der andere Name wird zum Symlink auf das Volume
#    (RENAME_PLAN 4.2 "the entrypoint maps the old mount"), in preflight -- der Baum liest ~/.cache/sglang hart kodiert.
# Regel: der neue Name gewinnt, wenn beide gesetzt sind; verschiedene Werte = CONFLICT-Zeile (laut), gleiche = still.
_ep_log() { printf '[htsglang-weg2 %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
EP_NAMES_NEW=()
EP_PRODUCT_NAMES=(PROFILE LINE STAND TREE MODE INSTRUMENTS JIT_MAX_JOBS ALLOW_EXPERIMENTAL EXPECT_GPUS MEMAVAIL_MIN_GIB
                  FORMAT_CHECK TRANSPORT USERDASH USERDASH_PORT USERDASH_BIND FORCE PROFILES_DIR
                  RIGDASH RIGDASH_PORT RIGDASH_BIND RIGDASH_LINE RIGDASH_TRUST_PROXY RIGDASH_MODEL_ROOTS)
ep_name() {   # ep_name <SUFFIX> [NEUER_NAME]: FLLIPER_<SUFFIX> (bzw. NEUER_NAME) -> HTSGLANG_<SUFFIX>
  local old=HTSGLANG_$1 new=${2:-FLLIPER_$1}
  [ -n "${!new+x}" ] || return 0
  if [ -n "${!old+x}" ] && [ "${!old}" != "${!new}" ]; then
    _ep_log "ENTRYPOINT names CONFLICT: $new='${!new}' and $old='${!old}' -- $new wins"
  fi
  export "$old=${!new}"
  EP_NAMES_NEW+=("$new")
  unset "$new"
}
ep_resolve_product_names() {   # die Namen, die dieses Skript selbst liest
  local s
  EP_PROFILE_SRC=default
  [ -z "${HTSGLANG_PROFILE+x}" ] || EP_PROFILE_SRC=HTSGLANG_PROFILE
  [ -z "${FLLIPER_PROFILE+x}" ] || EP_PROFILE_SRC=FLLIPER_PROFILE
  for s in "${EP_PRODUCT_NAMES[@]}"; do ep_name "$s"; done
  ep_name TAG FLLIPER_BOOT_TAG
}
ep_resolve_profile_names() {   # <profil-dir>: jedes HTSGLANG_<X>, das ein Profil liest (Profile sourcen einander -> alle)
  local s
  for s in $(grep -ho 'HTSGLANG_[A-Z0-9_]*[A-Z0-9]' "$1"/*.env 2>/dev/null | sort -u); do
    s=${s#HTSGLANG_}
    case " ${EP_PRODUCT_NAMES[*]} TAG " in *" $s "*) continue ;; esac
    case "$s" in REVISION*) continue ;; esac   # Image-ENV, kein Nutzerschalter
    ep_name "$s"
  done
}
EP_STATE_NEW=/var/lib/flliper; EP_STATE_OLD=/var/lib/htsglang; EP_PATHS=""
_ep_mounted() { [ ! -L "$1" ] && mountpoint -q "$1" 2>/dev/null; }   # ein Symlink (von ep_cache_link) ist kein Volume
_ep_on_mount() {   # _ep_on_mount <pfad> <basis>: pfad oder ein Elternteil bis einschliesslich basis ist ein Volume
  local p=$1
  while :; do
    _ep_mounted "$p" && return 0
    [ "$p" = "$2" ] && return 1
    p=${p%/*}; [ -n "$p" ] || return 1
  done
}
ep_state_path() {   # ep_state_path <SUFFIX> <neu-unterdir> <alt-unterdir>
  local new=$EP_STATE_NEW/$2 old=$EP_STATE_OLD/$3 nv=FLLIPER_PDFLIP_$1 ov=SGLANG_WEG2_$1 val src nm=0 om=0
  local nvv=${!nv:-} ovv=${!ov:-}
  [ "$ovv" != "$old" ] || ovv=""   # = Dockerfile-ENV-Default, keine Wahl des Aufrufers
  if [ -n "$nvv" ]; then
    val=$nvv; src=$nv
    if [ -n "$ovv" ] && [ "$ovv" != "$nvv" ]; then _ep_log "ENTRYPOINT paths CONFLICT: $nv='$nvv' and $ov='$ovv' -- $nv wins"; fi
  elif [ -n "$ovv" ]; then
    val=$ovv; src=$ov
  else
    _ep_on_mount "$new" "$EP_STATE_NEW" && nm=1
    _ep_on_mount "$old" "$EP_STATE_OLD" && om=1
    if [ "$nm$om" = 11 ]; then
      _ep_log "ENTRYPOINT paths CONFLICT: $new and $old are both on a volume -- $new wins, $old is NOT used"
      val=$new; src="volume, conflict"
    elif [ "$nm" = 1 ]; then val=$new; src=volume
    elif [ "$om" = 1 ]; then val=$old; src=volume
    else val=$old; src="default, no volume: container layer"
    fi
  fi
  export "$ov=$val" "$nv=$val"
  EP_PATHS+=" $1=$val ($src);"
}
EP_CACHE_NEW=$HOME/.cache/flliper; EP_CACHE_OLD=$HOME/.cache/sglang; EP_CACHE_LINK=""
ep_cache_dir() {   # setzt CACHE_DIR (Saat, Persistenz-Pruefung) und EP_CACHE_LINK ("<name> -> <volume>", fuer preflight)
  local nm=0 om=0
  _ep_mounted "$EP_CACHE_NEW" && nm=1
  _ep_mounted "$EP_CACHE_OLD" && om=1
  if [ "$nm$om" = 11 ]; then
    CACHE_DIR=$EP_CACHE_NEW; EP_CACHE_LINK=""
    _ep_log "ENTRYPOINT paths CONFLICT: $EP_CACHE_NEW and $EP_CACHE_OLD are both volumes -- $EP_CACHE_NEW wins for the entrypoint (seed files, persistence check); a tree of the sglang generation still reads $EP_CACHE_OLD"
    EP_PATHS+=" CACHE=$CACHE_DIR (volume, conflict);"
  elif [ "$nm" = 1 ]; then
    CACHE_DIR=$EP_CACHE_NEW; EP_CACHE_LINK="$EP_CACHE_OLD -> $EP_CACHE_NEW"; EP_PATHS+=" CACHE=$CACHE_DIR (volume);"
  elif [ "$om" = 1 ]; then
    CACHE_DIR=$EP_CACHE_OLD; EP_CACHE_LINK="$EP_CACHE_NEW -> $EP_CACHE_OLD"; EP_PATHS+=" CACHE=$CACHE_DIR (volume);"
  else
    CACHE_DIR=$EP_CACHE_OLD; EP_CACHE_LINK=""; EP_PATHS+=" CACHE=$CACHE_DIR (default, no volume: container layer);"
  fi
}
ep_cache_link() {   # preflight: der andere Cache-Name zeigt auf das Volume -- nie etwas ueberschreiben
  [ -n "$EP_CACHE_LINK" ] || return 0
  local from=${EP_CACHE_LINK%% -> *} to=${EP_CACHE_LINK##* -> } aside
  if [ -L "$from" ]; then
    [ "$(readlink "$from")" != "$to" ] || return 0
    rm -f "$from"
  elif [ -e "$from" ]; then
    # Image-Inhalt (z.B. rigmon/) wandert ins Volume, soweit dort nichts gleichnamiges liegt; das Verzeichnis selbst
    # wird nur beiseitegelegt (Container-Schicht), nie geloescht.
    [ -z "$(ls -A "$from" 2>/dev/null)" ] || cp -an "$from/." "$to/" 2>/dev/null \
      || _ep_log "WARN ENTRYPOINT paths: image content of $from not fully copied into $to"
    aside=$from.image-$(date -u +%m%d%H%M%S)
    mv "$from" "$aside" 2>/dev/null || { _ep_log "WARN ENTRYPOINT paths: $from not movable (volume below it?) -- stays, a tree of the sglang generation reads it, not $to"; return 0; }
  fi
  mkdir -p "${from%/*}" 2>/dev/null || true
  if ln -s "$to" "$from" 2>/dev/null; then _ep_log "ENTRYPOINT paths: $from -> $to (one cache directory under both names)"
  else _ep_log "WARN ENTRYPOINT paths: symlink $from -> $to failed -- a tree reading $from does not see the volume"; fi
}
# <<< ENTRYPOINT-NAMES

# Auftrag 1995 (Nutzer-Entscheid 05.10.: Profil-Editor ins Release): MODE=editor = nur der Profil-Editor (rigdash --edition release
# --editor-only) im Vordergrund, kein Server, keine GPU. Er schreibt Nutzerprofile (JSON) nach $FLLIPER_PROFILES_DIR
# (Standard /var/lib/flliper/profiles); danach startet FLLIPER_PROFILE=<name> mit MODE=weg2 genau dieses Profil.
# Das Paket liegt nur in Images, die mit RIGDASH_REV gebaut wurden (/opt/htsglang/rigdash).
# Test-Haken (nur fuer den hermetischen Test test_rigdash_editor_1995.sh): RIGDASH_DIR, RIGDASH_HOME (statt /opt/htsglang/rigdash, /opt/htsglang).
if [ "${MODE:-}" = "editor" ]; then
  ep_resolve_product_names
  RIGDASH_DIR=${RIGDASH_DIR:-/opt/htsglang/rigdash}
  if [ ! -f "$RIGDASH_DIR/entrypoint_rigdash.sh" ]; then
    printf '[htsglang-editor %s] REFUSED EDITOR: dieses Image traegt den Profil-Editor nicht (%s fehlt; Bau mit RIGDASH_REV)\n' "$(date -u +%H:%M:%SZ)" "$RIGDASH_DIR" >&2
    exit 3
  fi
  PY=${SGLANG_WEG2_VENV:-/opt/venv}/bin/python
  HOME_DIR=${RIGDASH_HOME:-/opt/htsglang}; PROFILE_DIR=$HOME_DIR/profiles; FRONT_PORT=30030
  : "${HTSGLANG_PROFILES_DIR:=/var/lib/flliper/profiles}"
  export PY HOME_DIR PROFILE_DIR FRONT_PORT HTSGLANG_PROFILES_DIR RIGDASH_DIR
  exec "$RIGDASH_DIR/entrypoint_rigdash.sh" "$@"
fi

# fLLiper (RENAME_PLAN 8.13 Schritt 1, FL5 26.09.): MODE=pdflip ist derselbe Modus wie MODE=weg2.
if [ "${MODE:-server}" != "weg2" ] && [ "${MODE:-server}" != "pdflip" ]; then
  # Zwei Code-Staende im Image (25.09.): der August-Server-/Planner-Modus laeuft auf dem 27B-Stand (HTSGLANG_STAND=nf
  # waehlt den NF-Stand); kein sglang in site-packages, also PYTHONPATH wie im weg2-Modus.
  ep_name STAND
  _st=${HTSGLANG_STAND:-27b}
  if [ -d "/opt/htsglang/src-$_st/python/sglang" ] || [ -d "/opt/htsglang/src-$_st/python/flliper" ]; then export PYTHONPATH=/opt/htsglang/src-$_st/python; fi
  exec /usr/local/bin/htsglang-entrypoint.sh "$@"
fi

# fLLiper Schritt 5 (EP 26.09.): FLLIPER_* -> HTSGLANG_* (Block ENTRYPOINT-NAMES oben), VOR dem ersten Leser.
ep_resolve_product_names
HOME_DIR=/opt/htsglang
VENV=${SGLANG_WEG2_VENV:-/opt/venv}
PY=$VENV/bin/python
TREE=${HTSGLANG_TREE:-/opt/htsglang/src}
PROFILE_DIR=$HOME_DIR/profiles
BUILD_INFO=$HOME_DIR/BUILD_INFO.json
JIT_REPORT=$HOME_DIR/JIT_PREBUILD.json
ARB_SEED=$HOME_DIR/arb-seed
# Stufe B (Commit bae049a3b4): der Launcher liest diese Pfade aus der Umgebung. fLLiper Schritt 5: Variable
# (FLLIPER_PDFLIP_* vor SGLANG_WEG2_*) oder Volume (/var/lib/flliper vor /var/lib/htsglang), sonst /var/lib/htsglang.
ep_state_path EVIDENCE_DIR evidence evidence
ep_state_path GPU_ARB arb arb
ep_state_path STORE_ROOT hicache hicache-weg2
ep_cache_dir
export SGLANG_WEG2_DEVTOOLS_DIR=${SGLANG_WEG2_DEVTOOLS_DIR:-/opt/htsglang/devtools}
export SGLANG_WEG2_VENV=$VENV
_TMS_BY_CALLER=${SGLANG_WEG2_TMS_OUT_DIR+1}; _TEXT_BY_CALLER=${TORCH_EXTENSIONS_DIR+1}
_BWCAP_BY_CALLER=${SGLANG_BARLINK_BUILD_WINDOW_CAP_S:-}; FIRST_BOOT_CACHE=0
export SGLANG_WEG2_TMS_OUT_DIR=${SGLANG_WEG2_TMS_OUT_DIR:-/opt/htsglang/tms}
# Rang-seitige Diagnose-Ziele, die sonst auf Rig-Pfade zeigen (barlink_abort_gate.py:679,
# barlink_capture_census.py:111-115).
export SGLANG_WEG2_PYSPY_DIR=${SGLANG_WEG2_PYSPY_DIR:-$SGLANG_WEG2_EVIDENCE_DIR}
export SGLANG_BARLINK_CAPTURE_CENSUS_DIR=${SGLANG_BARLINK_CAPTURE_CENSUS_DIR:-$SGLANG_WEG2_EVIDENCE_DIR/capture_census}
EVIDENCE_DIR=$SGLANG_WEG2_EVIDENCE_DIR
GPU_ARB=$SGLANG_WEG2_GPU_ARB
STORE_ROOT=$SGLANG_WEG2_STORE_ROOT
FRONT_PORT=30030

say()    { printf '[htsglang-weg2 %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
refuse() { say "REFUSED $1: $2"; exit 3; }
# PROFIL-EDITOR S1: Wert-Ablehnung (Kapazitaet, Schwelle, Beweisstand, Kalibrierung) -- mit FLLIPER_FORCE=1 (HTSGLANG_FORCE) uebergangen
# und laut gelistet (Boot-Log: FORCED-PAST <CODE> <Grund>); sonst genau wie refuse. NICHT hierher gehoeren Belegung der Karte, fehlendes
# Modell, kaputte Datei, nicht unterstuetzte Architektur: die bleiben refuse (weg2/refusals.py, Klasse nicht_forcebar).
refuse_value() { if [ "${HTSGLANG_FORCE:-0}" = 1 ]; then say "FORCED-PAST $2 $3"; return 0; fi; refuse "$1" "$3"; }   # <TAG> <CODE> <Text>
usage()  { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; }

# --- Untermodus -----------------------------------------------------------------
SUB=${HTSGLANG_MODE:-serve}
if [ $# -gt 0 ]; then
  case "$1" in
    serve|dryrun|preflight|selfcheck|version|bar1probe) SUB=$1; shift ;;
    bash|sh|/bin/bash|/bin/sh) exec "$@" ;;
    python|python3) shift; exec "$PY" "$@" ;;
    -h|--help|help) usage; exit 0 ;;
  esac
fi

if [ "$SUB" = "version" ]; then
  cat "$BUILD_INFO" 2>/dev/null || echo '{"BUILD_INFO": "fehlt"}'
  jq -c '{verdict, archs, flashinfer: [.flashinfer[]? | {dirkey, n: (.rows|length), built: ([.rows[]?|select(.status=="built")]|length)}], barlink: [.barlink.barlink[]? | {ext, status: .status[0:60]}], cpu_ext: [.cpu_ext.cpu_ext[]? | {ext, status: .status[0:60]}], tms: .tms.so, stages: [.stages[]?.only], problems}' "$JIT_REPORT" 2>/dev/null || true
  exit 0
fi

# --- 1. Profil -------------------------------------------------------------------
# Unbekannt oder ausdruecklich leer = harter Fehler, nie still das Default-Profil (EP 26.09.: README-Befehl mit
# FLLIPER_PROFILE bootete sonst still 27b).
[ "$EP_PROFILE_SRC" = default ] || [ -n "$HTSGLANG_PROFILE" ] || refuse PROFILE "$EP_PROFILE_SRC ist gesetzt, aber leer"
: "${HTSGLANG_PROFILE:=${HTSGLANG_LINE:-27b}}"
say "ENTRYPOINT names: profile from $EP_PROFILE_SRC -> '$HTSGLANG_PROFILE'"
say "ENTRYPOINT paths:$EP_PATHS"
# --- 1a. Code-Stand (Nutzer 25.09.: EIN Image mit zwei Code-Staenden) ---------------------------------------
# Die Profilfamilie waehlt den Stand: 27b* -> src-27b, nf* -> src-nf. Jeder Stand bringt seinen Baum, seinen Vorbau
# (TORCH_EXTENSIONS_DIR / TMS je Stand, gleiche Pfade wie im Dockerfile-Vorbau), seine ARB-Saat und sein BUILD_INFO.
# sglang liegt in keinem site-packages: der Stand wirkt ueber PYTHONPATH=<baum>/python -- dasselbe Importmodell, das
# der Launcher fuer Raenge und Front setzt (launcher.py:4123/5541/14799).
# PROFIL-EDITOR S1 (Auftrag 930): FLLIPER_PROFILE=<name> nennt ein Release-Profil (<profiles>/<name>.env, unveraendert) ODER ein vom
# Dashboard erstelltes Nutzerprofil (<state>/profiles/<name>.json, flliper.server/1). Gibt es kein Release-Profil dieses Namens, aber eine JSON-
# Datei, bestimmt deren "line" die Familie (der Name darf frei sein) und der Exporter macht sie unten zur .env (siehe PROFILE_FILE).
USER_PROFILE_DIR=${HTSGLANG_PROFILES_DIR:-/var/lib/flliper/profiles}   # FLLIPER_PROFILES_DIR; derselbe Ort wie im Dashboard
USER_PROFILE_JSON=""; USER_PROFILE_LINE=""
if [ ! -f "$PROFILE_DIR/${HTSGLANG_PROFILE}.env" ] && [ -f "$USER_PROFILE_DIR/${HTSGLANG_PROFILE}.json" ]; then
  USER_PROFILE_JSON=$USER_PROFILE_DIR/${HTSGLANG_PROFILE}.json
  USER_PROFILE_LINE=$(jq -r '.line // empty' "$USER_PROFILE_JSON" 2>/dev/null) || USER_PROFILE_LINE=""
  case "$USER_PROFILE_LINE" in 27b|nf) ;; *) refuse PROFILE "Nutzerprofil $USER_PROFILE_JSON traegt keine Linie (27b|nf) -- im Dashboard-Profil PROFILE_LINE setzen" ;; esac
  say "Nutzerprofil '$HTSGLANG_PROFILE' (JSON, Linie $USER_PROFILE_LINE) aus $USER_PROFILE_DIR"
fi
STAND=""
if [ -d "$HOME_DIR/src-27b" ] || [ -d "$HOME_DIR/src-nf" ]; then
  case "${USER_PROFILE_LINE:-$HTSGLANG_PROFILE}" in
    27b*) STAND=27b ;;
    nf*)  STAND=nf ;;
    *) refuse PROFILE "Profil '$HTSGLANG_PROFILE' gehoert zu keiner Familie (27b*|nf*) -- welcher Code-Stand, ist offen" ;;
  esac
  if [ -n "${HTSGLANG_TREE:-}" ] && [ "$HTSGLANG_TREE" != "$HOME_DIR/src-$STAND" ]; then
    refuse TREE "HTSGLANG_TREE=$HTSGLANG_TREE widerspricht der Profilfamilie $STAND ($HOME_DIR/src-$STAND)"
  fi
  TREE=$HOME_DIR/src-$STAND
  [ -d "$TREE/python/sglang" ] || [ -d "$TREE/python/flliper" ] || refuse TREE "Code-Stand $STAND fehlt im Image ($TREE)"
  BUILD_INFO=$HOME_DIR/BUILD_INFO-$STAND.json
  JIT_REPORT=$HOME_DIR/JIT_PREBUILD-$STAND.json
  ARB_SEED=$HOME_DIR/arb-seed-$STAND
  export HTSGLANG_LINE=$STAND HTSGLANG_TREE=$TREE PYTHONPATH=$TREE/python
  [ -n "$_TEXT_BY_CALLER" ] || export TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions/py312_cu130-$STAND
  [ -n "$_TMS_BY_CALLER" ] || export SGLANG_WEG2_TMS_OUT_DIR=$HOME_DIR/tms-$STAND
  STAND_REV=$(git -C "$TREE" rev-parse HEAD 2>/dev/null || echo unbekannt)
  say "Code-Stand $STAND: $TREE @ ${STAND_REV:0:10} (Profil $HTSGLANG_PROFILE, torch_extensions $TORCH_EXTENSIONS_DIR)"
  mkdir -p /tmp/htsglang && printf '%s %s\n' "$STAND" "$STAND_REV" > /tmp/htsglang/stand
fi
# fLLiper (RENAME_PLAN 8.13 Schritt 1, FL5 26.09.): der BAUM bestimmt die Paketnamen, kein Schalter. Alter Baum
# python/sglang + srt/weg2, umbenannter Baum python/flliper + srt/pdflip -- dasselbe Skript startet beide Generationen.
if [ -d "$TREE/python/flliper" ]; then PKG=flliper; PDF=pdflip; else PKG=sglang; PDF=weg2; fi
LAUNCHER_MOD=$PKG.srt.$PDF.launcher
PROFILE_FILE=$PROFILE_DIR/${HTSGLANG_PROFILE}.env
if [ -n "$USER_PROFILE_JSON" ]; then
  mkdir -p /tmp/htsglang
  PROFILE_FILE=/tmp/htsglang/user-profile-${HTSGLANG_PROFILE}.env
  PYTHONPATH="$TREE/python${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES="" "$PY" -m "$PKG.srt.$PDF.profile_json" render "$USER_PROFILE_JSON" -o "$PROFILE_FILE" \
    || refuse PROFILE "Nutzerprofil $USER_PROFILE_JSON laesst sich nicht in eine .env umsetzen (Exporter $PKG.srt.$PDF.profile_json fehlt im Code-Stand?)"
  say "Nutzerprofil -> $PROFILE_FILE (profile_json render)"
fi
[ -f "$PROFILE_FILE" ] || refuse PROFILE "unbekanntes Profil '$HTSGLANG_PROFILE' (vorhanden: $(cd "$PROFILE_DIR" && ls ./*.env 2>/dev/null | sed 's#^\./##; s/\.env$//' | tr '\n' ' '))"
# fLLiper Schritt 5: die Schalter, die Profile lesen (HTSGLANG_DRAFT, _P_CHUNK_POLICY, ...), auch als FLLIPER_*.
ep_resolve_profile_names "$PROFILE_DIR"
[ "${#EP_NAMES_NEW[@]}" -eq 0 ] || say "ENTRYPOINT names: read as HTSGLANG_*: ${EP_NAMES_NEW[*]}"
: "${HTSGLANG_TAG:=dkr${HTSGLANG_PROFILE//-/}$(date -u +%m%d%H%M%S)}"
export HTSGLANG_TAG
# F8: Instrument-Env im Release AUS, schaltbar. VOR dem Profil gesetzt, weil Profile den Schalter
# schon beim Sourcen lesen (nf.env: NF_ENV_*_INSTR in --env-p/--env-d).
: "${HTSGLANG_INSTRUMENTS:=0}"
case "$HTSGLANG_INSTRUMENTS" in 0|1) ;; *) refuse INSTRUMENTS "HTSGLANG_INSTRUMENTS='$HTSGLANG_INSTRUMENTS' (0|1)" ;; esac
export HTSGLANG_INSTRUMENTS
# Wer setzt SGLANG_PINNED_HOST_RESERVE_GIB? VOR dem Profil festhalten, ob der Aufrufer (-e) es gesetzt hat -- das Profil
# (z.B. nf.env: _form ... 2) setzt es sonst selbst, und pinned_reserve_for_cgroup soll die Quelle richtig nennen.
_PINNED_BY_CALLER=${SGLANG_PINNED_HOST_RESERVE_GIB+1}
_MAXJOBS_BY_CALLER=${MAX_JOBS+1}
_BAR1CONN_BY_CALLER=${SGLANG_WEG2_BAR1_CONNECT_S+1}
# shellcheck source=/dev/null
source "$PROFILE_FILE"
# Laufzeit-JIT darf den Container nicht sprengen (Operator 25.09., rc9-NVFP4 dkr27bnvfp4bar109251810: fp4_gemm_cutlass_sm120,
# 17 CUTLASS-TUs, nvcc code=137 im 76g-Deckel). Ohne MAX_JOBS ruft FlashInfer ninja OHNE -j (= Kerne+2, jit/cpp_ext.py
# run_ninja); torch_extensions nimmt MAX_JOBS ebenso. MAX_JOBS steht in keiner build.ninja und in keinem Cache-Schluessel,
# die Grenze erzwingt also keinen Neubau. FLASHINFER_NVCC_THREADS bleibt unberuehrt (es steht in den nvcc-Flags).
if [ -z "$_MAXJOBS_BY_CALLER" ] && [ -z "${MAX_JOBS:-}" ]; then
  export MAX_JOBS=${HTSGLANG_JIT_MAX_JOBS:-4}
  say "Laufzeit-JIT: MAX_JOBS=$MAX_JOBS (weder Aufrufer noch Profil setzen es; ohne Grenze baut ninja mit Kerne+2 Jobs -- CUTLASS-Module sprengen den Container-Deckel)"
elif [ -z "$_MAXJOBS_BY_CALLER" ]; then
  say "Laufzeit-JIT: MAX_JOBS=$MAX_JOBS (vom Profil)"
else
  say "Laufzeit-JIT: MAX_JOBS=$MAX_JOBS (vom Aufrufer)"
fi
[ "${PROFILE_LINE:-}" = "${HTSGLANG_LINE:-}" ] || refuse PROFILE "Profil-Linie '${PROFILE_LINE:-?}' != Image-Linie '${HTSGLANG_LINE:-?}' (27B und NF strikt getrennt: ein Image traegt genau eine Linie)"
# Profil-Stand (Release 25.09.): abgenommen = laeuft; experimentell = laeuft nur mit HTSGLANG_ALLOW_EXPERIMENTAL=1 und
# lauter Warnung; vorbereitet / geplant / Platzhalter = verweigert mit Grund. Altprofile ohne PROFILE_STATUS gelten als
# abgenommen, solange sie nicht PROFILE_PLACEHOLDER=1 tragen.
_pstat=${PROFILE_STATUS:-}
if [ -z "$_pstat" ]; then _pstat=abgenommen; [ "${PROFILE_PLACEHOLDER:-0}" = "1" ] && _pstat=platzhalter; fi
case "$_pstat" in
  abgenommen) ;;
  experimentell)
    [ "${HTSGLANG_ALLOW_EXPERIMENTAL:-0}" = "1" ] \
      || refuse_value PROFILE PROFIL-STATUS "Profil '$HTSGLANG_PROFILE' ist EXPERIMENTELL (${PROFILE_OWNER:-?}) -- nur mit HTSGLANG_ALLOW_EXPERIMENTAL=1"
    say "WARN: Profil '$HTSGLANG_PROFILE' ist EXPERIMENTELL (${PROFILE_OWNER:-?}) -- nicht abgenommen, auf eigenes Risiko" ;;
  formnachweis)   # NF 25.09. (nf-nvfp4-d): nativ gruen, Container-Abnahme ausstehend -- wie experimentell nur mit Freigabe
    [ "${HTSGLANG_ALLOW_EXPERIMENTAL:-0}" = "1" ] \
      || refuse_value PROFILE PROFIL-STATUS "Profil '$HTSGLANG_PROFILE' hat nur einen FORMNACHWEIS (nativ, ${PROFILE_OWNER:-?}) -- Container-Abnahme ausstehend; nur mit HTSGLANG_ALLOW_EXPERIMENTAL=1"
    say "WARN: Profil '$HTSGLANG_PROFILE' hat nur einen FORMNACHWEIS -- Container-Abnahme ausstehend, auf eigenes Risiko" ;;
  vorbereitet) refuse_value PROFILE PROFIL-STATUS "Profil '$HTSGLANG_PROFILE' ist VORBEREITET, nicht abgenommen (${PROFILE_OWNER:-?})" ;;
  geplant) refuse_value PROFILE PROFIL-STATUS "Profil '$HTSGLANG_PROFILE' ist GEPLANT (${PROFILE_OWNER:-?}) -- Form noch nicht geliefert" ;;
  *) refuse_value PROFILE PROFIL-STATUS "Profil '$HTSGLANG_PROFILE' ist ein PLATZHALTER (${PROFILE_OWNER:-Eigentuemer offen}) -- Form noch nicht geliefert" ;;
esac
[ "${#PROFILE_ARGS[@]}" -gt 0 ] || refuse PROFILE "Profil '$HTSGLANG_PROFILE' hat keine Launcher-Argumente (Form nicht geliefert)"
# D-only (NF 25.09., Launcher --d-only 1beabe6589): nur Gruppe D auf allen Karten, KEINE Front, Clients auf D direkt
# (PROFILE_SERVE_PORT, NF: 30032). Der Entrypoint haengt --d-only an, falls das Profil es nicht selbst traegt, und
# beaufsichtigt dann den D-Server statt der Front.
D_ONLY=${PROFILE_D_ONLY:-0}
case "$D_ONLY" in 0|1) ;; *) refuse PROFILE "PROFILE_D_ONLY='$D_ONLY' (0|1)" ;; esac
if [ "$D_ONLY" = 1 ]; then
  FRONT_PORT=${PROFILE_SERVE_PORT:-30032}
  _has_donly=0; for _a in "${PROFILE_ARGS[@]}"; do [ "$_a" = "--d-only" ] && _has_donly=1; done
  [ "$_has_donly" = 1 ] || { PROFILE_ARGS+=(--d-only); say "D-only: --d-only an den Launcher angehaengt (Profil setzt PROFILE_D_ONLY=1)"; }
  unset _has_donly _a
else
  FRONT_PORT=${PROFILE_SERVE_PORT:-$FRONT_PORT}
fi
[[ $FRONT_PORT =~ ^[0-9]+$ ]] || refuse PROFILE "PROFILE_SERVE_PORT='$FRONT_PORT' ist keine Portnummer"
mkdir -p /tmp/htsglang && printf '%s\n' "$FRONT_PORT" > /tmp/htsglang/serve_port   # fuer healthcheck.sh

_form() {   # _form VAR BESTFORM -- setzt Bestform, meldet eine explizite Abweichung laut
  local var=$1 best=$2
  if [ -n "${!var+x}" ] && [ "${!var}" != "$best" ]; then
    say "FORM-OVERRIDE $var=${!var} (Bestform $best) -- per docker run -e gesetzt"
  else
    export "$var=$best"
  fi
}
profile_form_env
if [ "$HTSGLANG_INSTRUMENTS" = "1" ]; then profile_instr_env; say "Instrumente AN (HTSGLANG_INSTRUMENTS=1)"; fi

# --- 2. Modellformat ---------------------------------------------------------------
detect_format() {   # detect_format <modellpfad> -> int8|int4|int4-mixed|fp8|nvfp4-modelopt|nvfp4-ct|gguf|bf16|...
  # Am Rig geprueft (24.09.): INT8-gdncov-vocabembed=int8, Flash-Next-Minachist=int4-mixed, 27B-FP8=fp8,
  # 27B-NVFP4-RadixArk und Flash-Next-NVFP4-nvidia=nvfp4-modelopt, 27B-NVFP4 (unsloth)=nvfp4-ct,
  # 27B-GGUF-unsloth (Verzeichnis MIT config.json) und Flash-Next-GGUF (gguf nur in Unterordnern)=gguf.
  "$PY" - "$1" <<'EOF'
import glob, json, os, sys
p = sys.argv[1]
if os.path.isfile(p) and p.endswith(".gguf"):
    print("gguf"); sys.exit()
# GGUF-Verzeichnis: *.gguf (Ebene 0-2), keine *.safetensors. Eine config.json darf daneben liegen
# (unsloth legt die HF-Konfiguration fuer Tokenizer/Architektur bei), sie macht es nicht zu bf16.
if os.path.isdir(p) and not glob.glob(os.path.join(p, "*.safetensors")) and (
        glob.glob(os.path.join(p, "*.gguf")) or glob.glob(os.path.join(p, "*", "*.gguf"))):
    print("gguf"); sys.exit()
cfg = json.load(open(os.path.join(p, "config.json")))
q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
m = str(q.get("quant_method") or "").lower()
kinds = {(str((g.get("weights") or {}).get("type")), (g.get("weights") or {}).get("num_bits"))
         for g in (q.get("config_groups") or {}).values()}
hq = os.path.join(p, "hf_quant_config.json")
if m == "fp8":
    print("fp8")
elif m == "modelopt" or os.path.exists(hq):
    # modelopt-NVFP4 (nvidia, RadixArk; hf_quant_config.json, MIXED_PRECISION) ist ein anderer
    # Lader als compressed-tensors-NVFP4 (unsloth) -> eigene Namen, das Profil nennt genau einen.
    algo = str(q.get("quant_algo") or "").upper()
    print("nvfp4-modelopt" if ("float", 4) in kinds or "FP4" in algo else ("fp8" if "FP8" in algo else "modelopt:" + algo.lower()))
elif m == "compressed-tensors":
    if ("float", 4) in kinds: print("nvfp4-ct")
    elif kinds == {("int", 8)}: print("int8")
    elif kinds == {("int", 4)}: print("int4")
    elif ("int", 4) in kinds: print("int4-mixed")
    elif kinds and all(k == ("float", 8) for k in kinds): print("fp8")
    else: print("compressed-tensors:" + ",".join(f"{t}{b}" for t, b in sorted(kinds, key=str)))
elif m in ("awq", "gptq", "auto-round", "autoround", "awq_marlin", "gptq_marlin"):
    print("int4")
else:
    print(m or "bf16")
EOF
}

# --- 3. Umgebung saeubern ------------------------------------------------------------
if [ -n "${FLASHINFER_CUDA_ARCH_LIST+x}" ]; then
  say "WARN: FLASHINFER_CUDA_ARCH_LIST='$FLASHINFER_CUDA_ARCH_LIST' entfernt -- vor dem flashinfer-Import gesetzt verschoebe es das JIT-Cache-Verzeichnis (0.6.14/120f bzw. /86); der Server setzt es selbst nach dem Import (utils/common.py:1547)"
  unset FLASHINFER_CUDA_ARCH_LIST
fi
if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  # Das August-Image setzt 7 Architekturen fuer den Server-Modus. Im weg2-Modus gewinnt
  # sonst diese Liste ueber die Gruppen-Union (barlink_device.py:652-657) und keiner der
  # vorgebauten *_cuda_86_120-Exts passt -> minutenlanger nvcc-Bau im Boot.
  say "HINWEIS: TORCH_CUDA_ARCH_LIST='$TORCH_CUDA_ARCH_LIST' fuer den weg2-Modus entfernt (Gruppen-Union entscheidet)"
  unset TORCH_CUDA_ARCH_LIST
fi
case "${PYTORCH_CUDA_ALLOC_CONF:-}" in
  *expandable_segments*) refuse ALLOC_CONF "PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF': torch_memory_saver verweigert expandable_segments (launcher.py:5515)" ;;
esac

# --- 4. Preflight ------------------------------------------------------------------
GIB=$((1024 * 1024 * 1024))
BDFS=()

# Namens-Gate (bis HW-GENERIC 1002 das einzige): PROFILE_GPUS="<NVML-Name>=<Anzahl>;..." exakt gegen nvidia-smi.
# Bleibt UNVERAENDERT der Rueckfall fuer Baeume ohne card_identity.py und Profile ohne PROFILE_CARD_COUNT.
gpu_name_gate() {
  local rows=$1 want name n have w
  IFS=';' read -r -a want <<< "$PROFILE_GPUS"
  for w in "${want[@]}"; do
    name=${w%=*}; n=${w##*=}
    have=$(awk -F', ' -v nm="$name" '$2==nm' <<< "$rows" | wc -l)
    [ "$have" = "$n" ] || refuse GPU "Profil '$HTSGLANG_PROFILE' erwartet ${n}x '$name', sichtbar ${have}x (Rig-Profil, auf dieses Karteninventar kalibriert)"
  done
}

# HW-GENERIC 1002 (Nutzer 02.10.: "die software soll fuer jede hardware laufen ... aktuell halt 'nur' sm86 und sm120"):
# das Gate aus NVML-EIGENSCHAFTEN statt NAMEN -- $PKG.srt.$PDF.card_identity (CLI) mit dem Image-Python gegen den Baum:
#   rc 0 = Arch, Anzahl und (mit PROFILE_INVENTORY) Kalibrier-Inventar passen; die Kartentabelle (Ordinal-Reihenfolge,
#          Klasse je Karte) steht im Log;
#   rc 3 = HW-ARCH (Karte ausserhalb sm_86/sm_89/sm_120; sm_89 ist seit SM89-1002
#          angenommen -- es landet auf rc 4 HW-UNCALIBRATED) oder HW-COUNT (nicht PROFILE_CARD_COUNT Karten);
#   rc 4 = HW-UNCALIBRATED (die Positionsvektoren des Profils sind auf einem anderen Inventar gemessen).
# HTSGLANG_EXPECT_GPUS != 1 bleibt die Operator-Ausnahme wie bisher: HW-COUNT, HW-UNCALIBRATED und ein scheiterndes
# CLI werden zur WARNUNG. HW-ARCH verweigert immer -- das Image traegt fuer andere Archs keine Kernel, der Launcher
# (resolve_cards -> card_identity.arch_gate) verweigerte dieselbe Karte sonst erst nach dem Preflight.
gpu_identity_gate() {
  local args=(--expect-count "$PROFILE_CARD_COUNT") out rc=0 line msg="" esc=0
  [ "${HTSGLANG_EXPECT_GPUS:-1}" = "1" ] || esc=1
  [ -z "${PROFILE_INVENTORY:-}" ] || args+=(--inventory "$PROFILE_INVENTORY")
  out=$(CUDA_VISIBLE_DEVICES="" PYTHONPATH="$TREE/python${PYTHONPATH:+:$PYTHONPATH}" \
        timeout 120 "$PY" -m "$PKG.srt.$PDF.card_identity" "${args[@]}" 2>&1) || rc=$?
  say "GPU-Gate card_identity (${args[*]}): rc=$rc"
  while IFS= read -r line; do [ -z "$line" ] || say "  $line"; done <<< "$out"
  while IFS= read -r line; do
    case $line in
      "refuse HW-"*) msg=${line#refuse }; break ;;
      HW-UNCALIBRATED:*) msg=$line; break ;;
    esac
  done <<< "$out"
  case $rc in
    0) say "GPU-Gate: Arch sm_86/sm_89/sm_120, ${PROFILE_CARD_COUNT} Karten${PROFILE_INVENTORY:+, Kalibrier-Inventar [$PROFILE_INVENTORY]} -- passt" ;;
    3) [ -n "$msg" ] || msg="card_identity rc=3 ohne benannte Meldung: ${out##*$'\n'}"
       case $msg in
         HW-COUNT:*) if [ "$esc" = 1 ]; then say "WARN: $msg -- uebergangen (HTSGLANG_EXPECT_GPUS=${HTSGLANG_EXPECT_GPUS}, Operator-Ausnahme)"; return 0; fi
                     if [ "${HTSGLANG_FORCE:-0}" = 1 ]; then say "FORCED-PAST HW-COUNT $msg"; return 0; fi ;;
       esac
       refuse GPU "$msg" ;;
    4) [ -n "$msg" ] || msg="HW-UNCALIBRATED: card_identity rc=4 ohne benannte Meldung: ${out##*$'\n'}"
       if [ "${HTSGLANG_FORCE:-0}" = 1 ]; then say "FORCED-PAST HW-UNCALIBRATED $msg"; return 0; fi
       if [ "$esc" = 1 ]; then say "WARN: $msg -- uebergangen (HTSGLANG_EXPECT_GPUS=${HTSGLANG_EXPECT_GPUS}, Operator-Ausnahme)"; return 0; fi
       refuse UNCALIBRATED "$msg (Operator-Ausnahme auf eigenes Risiko: HTSGLANG_EXPECT_GPUS=0)" ;;
    *) if [ "$esc" = 1 ]; then say "WARN: card_identity scheitert (rc=$rc) -- uebergangen (HTSGLANG_EXPECT_GPUS=${HTSGLANG_EXPECT_GPUS})"; return 0; fi
       refuse GPU "card_identity scheitert (rc=$rc): ${out##*$'\n'}" ;;
  esac
}

preflight_gpus() {
  command -v nvidia-smi >/dev/null || refuse GPU "nvidia-smi fehlt im Container (Treiber-Userspace nicht eingebunden: --gpus all bzw. CDI)"
  local rows
  rows=$(nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader 2>&1) \
    || refuse GPU "nvidia-smi scheitert: $rows"
  say "GPU-Inventar (NVML):"
  while IFS= read -r r; do say "  $r"; done <<< "$rows"
  DRIVER_SEEN=$(awk -F', ' 'NR==1{print $6}' <<< "$rows")
  local bus dom rest
  while IFS= read -r bus; do
    dom=${bus%%:*}; rest=${bus#*:}
    BDFS+=("$(printf '%04x:%s' "$((16#$dom))" "${rest,,}")")
  done < <(awk -F', ' '{print $4}' <<< "$rows")
  local ci=$TREE/python/$PKG/srt/$PDF/card_identity.py
  # SM89-1002: PROFILE_GPUS ist KEIN Namens-Riegel mehr. Wo card_identity.py im Baum
  # liegt, gate't NVML-EIGENSCHAFT (Arch, Anzahl, optional Kalibrier-Inventar);
  # PROFILE_GPUS liefert dann nur die KARTENANZAHL (Summe der =N-Teile), die Namen
  # werden geloggt, nie verglichen -- ein Ada-Rig (oder jede fremde Bestueckung)
  # faellt nicht mehr an Kartennamen, sondern misst sich an den Eigenschaften
  # (HW-COUNT / HW-UNCALIBRATED; HW-ARCH nur ausserhalb sm86/89/120).
  # Das Namens-Gate bleibt reiner Rueckfall fuer Baeume OHNE card_identity.py.
  if [ -f "$ci" ]; then
    if [ -z "${PROFILE_CARD_COUNT:-}" ] && [ -n "${PROFILE_GPUS:-}" ]; then
      local parts pp n=0
      IFS=';' read -r -a parts <<< "$PROFILE_GPUS"
      for pp in "${parts[@]}"; do
        if [[ ${pp##*=} =~ ^[0-9]+$ ]]; then n=$((n + ${pp##*=})); fi
      done
      if [ "$n" -gt 0 ]; then
        PROFILE_CARD_COUNT=$n
        say "GPU-Gate: PROFILE_CARD_COUNT fehlt -- aus PROFILE_GPUS gezaehlt: $n (SM89-1002: Namen werden nicht verglichen)"
      fi
    fi
    if [ -n "${PROFILE_CARD_COUNT:-}" ]; then
      gpu_identity_gate
    else
      say "WARN: weder PROFILE_CARD_COUNT noch PROFILE_GPUS -- GPU-Gate ohne Erwartung, uebersprungen"
    fi
  elif [ "${HTSGLANG_EXPECT_GPUS:-1}" = "1" ] && [ -n "${PROFILE_GPUS:-}" ]; then
    say "GPU-Gate: Baum ohne $ci (aelterer Stand) -- Rueckfall auf das Namens-Gate PROFILE_GPUS"
    gpu_name_gate "$rows"
  fi
}

preflight_shm() {
  local size
  size=$(df -B1 --output=size /dev/shm | tail -1 | tr -d ' ')
  [ "$size" -ge $((PROFILE_SHM_MIN_GIB * GIB)) ] \
    || refuse_value SHM SHM "/dev/shm = $((size / GIB)) GiB < ${PROFILE_SHM_MIN_GIB} GiB -- docker run --shm-size (27B: 48g, NF: 16g); nie --ipc=host (Launcher-#1217 saehe fremde Halter)"
  say "/dev/shm $((size / GIB)) GiB (Minimum ${PROFILE_SHM_MIN_GIB})"
}

preflight_memory() {
  local total avail
  total=$(awk '/^MemTotal:/{print $2}' /proc/meminfo); avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  say "meminfo: MemTotal $((total / 1048576)) GiB, MemAvailable $((avail / 1048576)) GiB; cgroup memory.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo ?)"
  # HTSGLANG_MEMAVAIL_MIN_GIB (25.09.): ein Aufrufer mit eigener, strengerer Haus-Absicherung (host_acceptance.sh:
  # Host-Schwelle + Container-Deckel + --oom-score-adj) setzt die Untergrenze selbst; sonst gilt die Profil-Empfehlung.
  local min=${HTSGLANG_MEMAVAIL_MIN_GIB:-${PROFILE_MEMAVAIL_MIN_GIB:-40}} src=Profil
  [ -n "${HTSGLANG_MEMAVAIL_MIN_GIB:-}" ] && src=HTSGLANG_MEMAVAIL_MIN_GIB
  if [ $((avail / 1048576)) -lt "$min" ]; then
    [ "$SUB" = "dryrun" ] && say "WARN: MemAvailable < $min GiB ($src; nur Trockenlauf)" \
      || refuse_value MEMORY MEMAVAIL "MemAvailable $((avail / 1048576)) GiB < $min GiB ($src)"
  fi
  pinned_reserve_for_cgroup
}

# NF-Abnahme 25.09. (Operator-Entscheid b): pinned_host_budget haelt 10 GiB "OS-Reserve" frei, gemessen gegen den
# cgroup-Deckel minus nonreclaim. Unter einem Container-Deckel ist das doppelt gezaehlt -- den Host schuetzen dort
# schon Deckel, --oom-score-adj und die Host-Invariante (Deckel + 4 GiB <= MemAvailable) -- und es verweigerte bei
# NF-RC2 unter 82g die HiCache-Lesepuffer an der Spitze (8,73 GB frei - 10,74 GB Reserve = 0) -> W53/413.
# Darum: mit erkanntem cgroup-Deckel SGLANG_PINNED_HOST_RESERVE_GIB=2, sonst nichts (Code-Default 10 = nativ).
# Ein vom Nutzer gesetzter Wert gilt immer. Wirksam ab den Baeumen mit dem Knopf (NF-RC2.1, 27B-RC7b); aeltere
# Baeume ignorieren die Variable.
pinned_reserve_for_cgroup() {
  local mx
  mx=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
  if [ -n "${_PINNED_BY_CALLER:-}" ]; then
    say "pinned-Reserve: SGLANG_PINNED_HOST_RESERVE_GIB=${SGLANG_PINNED_HOST_RESERVE_GIB:-} (vom Aufrufer per -e gesetzt, gilt)"
  elif [ -n "${SGLANG_PINNED_HOST_RESERVE_GIB:-}" ]; then
    say "pinned-Reserve: SGLANG_PINNED_HOST_RESERVE_GIB=${SGLANG_PINNED_HOST_RESERVE_GIB} (vom Profil gesetzt, cgroup memory.max=$mx)"
  elif [[ $mx =~ ^[0-9]+$ ]]; then
    # 29.09. (27B-Entscheid, Grundgesetz: keine Pauschal-Reserve): KEIN fester Wert mehr. Unter dem Deckel setzt
    # der Launcher die gemessene Ledger-Marge (SGLANG_PINNED_HOST_RESERVE_LEDGER_GIB, desk/nf-pinned-reserve-env-0929);
    # die Launcher-Zeile WEG2-HOST PINNED-RESERVE nennt Wert und Quelle. Ein -e-Wert gilt weiter.
    say "pinned-Reserve: cgroup-Speicherdeckel erkannt ($((mx / GIB)) GiB) -> kein fester Wert; der Launcher setzt die gemessene Ledger-Marge (eigener Wert per -e hat Vorrang)"
  else
    say "pinned-Reserve: kein cgroup-Speicherdeckel (memory.max=$mx) -> Code-Default (10 GiB)"
  fi
}

preflight_tree() {
  local head
  head=$(git -C "$TREE" rev-parse HEAD 2>/dev/null) || refuse TREE "$TREE ist kein git-Baum (Launcher-Stempel launcher.py:12418, Linien-Identitaet line_identity.py:82)"
  # Zwei Staende: die Soll-Revision des gewaehlten Stands (Image-ENV HTSGLANG_REVISION_27B / _NF).
  local want=${HTSGLANG_REVISION:-}
  case "${STAND:-}" in 27b) want=${HTSGLANG_REVISION_27B:-} ;; nf) want=${HTSGLANG_REVISION_NF:-} ;; esac
  [ "$head" = "$want" ] || refuse TREE "HEAD $head != Image-Revision ${want:-?}${STAND:+ (Stand $STAND)}"
  [ -z "$(git -C "$TREE" status --porcelain)" ] || refuse TREE "Baum nicht sauber: $(git -C "$TREE" status --porcelain | head -3 | tr '\n' ' ')"
  say "Baum $TREE @ ${head:0:10} sauber"
}

preflight_model() {
  [ -e "$PROFILE_MODEL" ] || refuse MODEL "Modell fehlt: $PROFILE_MODEL (read-only unter demselben Pfad mounten)"
  [ -z "${PROFILE_DRAFT:-}" ] || [ -e "$PROFILE_DRAFT" ] || refuse MODEL "Draft fehlt: $PROFILE_DRAFT"
  local req
  for req in "${PROFILE_REQUIRED_PATHS[@]:-}"; do   # z.B. GGUF: das Tokenizer-Verzeichnis (--tokenizer-path)
    [ -z "$req" ] || [ -e "$req" ] || refuse MODEL "vom Profil verlangter Pfad fehlt: $req"
  done
  local fmt
  fmt=$(detect_format "$PROFILE_MODEL" 2>&1) || refuse MODEL "Formaterkennung scheitert: $fmt"
  say "Modellformat erkannt: $fmt ($PROFILE_MODEL)"
  if [ -n "${PROFILE_FORMAT:-}" ] && [ "$fmt" != "$PROFILE_FORMAT" ] && [ "${HTSGLANG_FORMAT_CHECK:-1}" = "1" ]; then
    refuse MODEL "Profil '$HTSGLANG_PROFILE' erwartet Format '$PROFILE_FORMAT', gemountet ist '$fmt'"
  fi
}

preflight_state() {
  local d
  for d in "$EVIDENCE_DIR" "$STORE_ROOT" "$GPU_ARB/weg2" "$SGLANG_WEG2_TMS_OUT_DIR"; do
    mkdir -p "$d" 2>/dev/null || true
    { touch "$d/.w" && rm -f "$d/.w"; } 2>/dev/null || refuse STATE "nicht beschreibbar: $d"
  done
  # ARB-Saat (PROBE_RING, calib, corridor-Samples, census): nur fehlende Dateien ergaenzen, nie ueberschreiben.
  if [ -d "$ARB_SEED" ]; then
    (cd "$ARB_SEED" && find . -type f -print0) | while IFS= read -r -d '' f; do
      [ -e "$GPU_ARB/$f" ] || { mkdir -p "$(dirname "$GPU_ARB/$f")"; cp -p "$ARB_SEED/$f" "$GPU_ARB/$f"; }
    done
  fi
  say "Zustand: evidence=$EVIDENCE_DIR ($(find "$EVIDENCE_DIR" -maxdepth 1 -name '*.D.log' 2>/dev/null | wc -l) D-Logs als Kalibrierquellen), arb=$GPU_ARB, store=$STORE_ROOT"
  if [ -n "${PROFILE_TMPFS_STORE:-}" ]; then
    [ "$(stat -f -c %T "$PROFILE_TMPFS_STORE" 2>/dev/null)" = "tmpfs" ] \
      || refuse_value STORE STORE "$PROFILE_TMPFS_STORE ist kein tmpfs (cudaHostRegister auf ZFS-mmap -> cudaError 1) -- --mount type=tmpfs,dst=$PROFILE_TMPFS_STORE,tmpfs-size=${PROFILE_TMPFS_GIB:-72}g"
    local sz
    sz=$(df -B1 --output=size "$PROFILE_TMPFS_STORE" | tail -1 | tr -d ' ')
    [ "$sz" -ge $(( ${PROFILE_TMPFS_GIB:-72} * GIB )) ] || refuse_value STORE STORE "$PROFILE_TMPFS_STORE = $((sz / GIB)) GiB < ${PROFILE_TMPFS_GIB:-72} GiB"
    clean_tmpfs_store "$PROFILE_TMPFS_STORE"
  fi
  seed_profile_files
}

# NF 25.09. (e): ein zweiter serve im selben Container faende im tmpfs-Store sonst ~39 GiB Altlast (RAM, zaehlt in den
# Container-Deckel). Geraeumt wird nur, wenn in diesem Container kein weg2-Prozess mehr lebt; sonst verweigern, statt
# einem laufenden Boot den Store unter den Fuessen wegzuziehen. Bash-Regex statt grep-Pipe (kein SIGPIPE unter pipefail).
clean_tmpfs_store() {
  local st=$1 c cmd live=0 old
  # nur ECHTE Prozesse: Prozesstitel sglang::scheduler*/detokenizer* oder ein Python-Interpreter mit -m ...launcher.
  # Wer die Namen nur ERWAEHNT (pgrep/grep/inotifywait/bash -c eines Waechters) blockiert nicht (25.09. im Test gesehen).
  # fLLiper (RENAME_PLAN 8.11, FL3 26.09.): alte UND neue Namen -- flliper::, -m flliper.srt.pdflip.launcher.
  local re='^(sglang|flliper)::(scheduler|detokenizer)|^[^ ]*python[0-9.]* (.* )?-m (sglang\.srt\.weg2|flliper\.srt\.pdflip)\.launcher( |$)'
  [ -n "$(ls -A "$st" 2>/dev/null)" ] || return 0
  for c in /proc/[0-9]*/cmdline; do
    [ "$c" = "/proc/$$/cmdline" ] && continue
    cmd=$(tr '\0' ' ' < "$c" 2>/dev/null) || continue
    if [[ $cmd =~ $re ]]; then live=1; break; fi
  done
  [ "$live" = 0 ] || refuse STORE "$st nicht leer und ein weg2-Prozess lebt noch in diesem Container -- erst beenden"
  old=$(du -sb "$st" 2>/dev/null | cut -f1) || old=0
  find "$st" -mindepth 1 -delete || refuse STORE "$st: Altlast nicht loeschbar"
  say "Store $st geraeumt: $(( ${old:-0} / 1048576 )) MiB Rest eines frueheren Laufs in diesem Container"
}

# NF 25.09. (c): Saat-Dateien des Profils (im Image unter /opt/htsglang/profiles/<linie>/, Liste profiles/<linie>.assets)
# an ihre Laufzeit-Orte -- NUR wenn sie dort fehlen: ein gemountetes evidence-/Cache-Verzeichnis mit eigenen Messungen
# gewinnt immer. PROFILE_SEED_EVIDENCE -> $EVIDENCE_DIR (weg2_measured_record.json), PROFILE_SEED_SGLANG_CACHE ->
# ~/.cache/sglang (card_probe.CACHE_DIR: phase_footprint-*.json, card_library.json).
seed_profile_files() {
  local f t n=0 have=0 cache=$CACHE_DIR   # fLLiper Schritt 5: ~/.cache/flliper oder ~/.cache/sglang (ep_cache_dir)
  for f in "${PROFILE_SEED_EVIDENCE[@]:-}"; do
    [ -n "$f" ] || continue
    [ -f "$f" ] || refuse STATE "Saat-Datei des Profils fehlt im Image: $f"
    t="$EVIDENCE_DIR/$(basename "$f")"
    if [ -e "$t" ]; then have=$((have + 1)); else cp -p "$f" "$t"; n=$((n + 1)); fi
  done
  for f in "${PROFILE_SEED_SGLANG_CACHE[@]:-}"; do
    [ -n "$f" ] || continue
    [ -f "$f" ] || refuse STATE "Saat-Datei des Profils fehlt im Image: $f"
    mkdir -p "$cache"
    t="$cache/$(basename "$f")"
    if [ -e "$t" ]; then have=$((have + 1)); else cp -p "$f" "$t"; n=$((n + 1)); fi
  done
  [ $((n + have)) -eq 0 ] || say "Profil-Saat: $n Datei(en) gesetzt, $have schon vorhanden (nie ueberschrieben)"
}

BAR1_OK=1; BAR1_WHY=()
preflight_bar1() {
  local params regkeys cap_eff bdf want_drv hdr
  params=$(grep -E '^RegistryDwords:' /proc/driver/nvidia/params 2>/dev/null || true)
  regkeys=${params#RegistryDwords: }
  case "$regkeys" in *RMSmallBarP2PPeerBar1=1*) ;; *) BAR1_OK=0; BAR1_WHY+=("Regkey RMSmallBarP2PPeerBar1=1 fehlt (gepatchter smallbar-Treiber)");; esac
  cap_eff=$(awk '/^CapEff:/{print $2}' /proc/self/status)
  case "$regkeys" in
    *PeerMappingOverride=1*) ;;
    *) if (( (16#$cap_eff >> 21) & 1 )); then :; else BAR1_OK=0; BAR1_WHY+=("weder PeerMappingOverride=1 noch CAP_SYS_ADMIN (barlink_bar1.py:3075-3100)"); fi ;;
  esac
  if [ -c /dev/dmabuf_holder ] && [ -r /dev/dmabuf_holder ] && [ -w /dev/dmabuf_holder ]; then :; else
    BAR1_OK=0; BAR1_WHY+=("/dev/dmabuf_holder fehlt/nicht rw (--device /dev/dmabuf_holder)"); fi
  for bdf in "${BDFS[@]}"; do
    [ -w "/sys/bus/pci/devices/$bdf/resource1_wc" ] || { BAR1_OK=0; BAR1_WHY+=("/sys/bus/pci/devices/$bdf/resource1_wc nicht beschreibbar (-v /sys/devices:/sys/devices, nie ganz /sys; Host-Rechte 0666)"); }
  done
  hdr=${SGLANG_BARLINK_BAR1_NV_SOURCE:-/opt/nvidia-open-595}/src/common/sdk/nvidia/inc/nvos.h
  [ -f "$hdr" ] || { BAR1_OK=0; BAR1_WHY+=("NV-Header fehlen unter ${SGLANG_BARLINK_BAR1_NV_SOURCE:-/opt/nvidia-open-595} (Image mit WITH_NV_HEADERS=0 gebaut; Header des LAUFENDEN Treibers mounten oder mit WITH_NV_HEADERS=1 bauen)"); }
  want_drv=$(jq -r '.driver_expected // empty' "$BUILD_INFO" 2>/dev/null || true)
  if [ -f "$hdr" ] && [ -n "$want_drv" ] && [ "${DRIVER_SEEN:-}" != "$want_drv" ] && [ "${SGLANG_BARLINK_BAR1_NV_SOURCE:-/opt/nvidia-open-595}" = "/opt/nvidia-open-595" ]; then
    BAR1_OK=0; BAR1_WHY+=("Treiber ${DRIVER_SEEN:-?} != Header-Stand im Image $want_drv (UAPI versionsgebunden, barlink_bar1_ext.py:14-21)")
  fi
  if [ "$BAR1_OK" = "1" ]; then say "BAR1-Kette vollstaendig (Regkeys: $regkeys)"; else say "BAR1-Kette unvollstaendig: ${BAR1_WHY[*]}"; fi
}

# JIT-/Autotune-Caches (Operator 25.09., Risiko F): FlashInfer-JIT und Autotuning kosten je (Shape, M) 5-29 s beim ersten
# Aufruf; der sglang-Autotune-Cache (SGLANG_CACHE_DIR/flashinfer/autotune), der FlashInfer-Cache (FLASHINFER_WORKSPACE_BASE
# -> ~/.cache/flashinfer: JIT, cubins, autotune) und der CuTe-DSL-Cache muessen auf PERSISTENTEN Volumes liegen, sonst
# zahlt jeder Neustart das von vorn. Liegt ein Cache in der Container-Schicht: laute Warnung (kein Abbruch -- ein
# Wegwerf-Lauf darf kalt sein). JE STAND getrennt: der Stand stempelt jedes gemountete Cache-Volume (.htsglang-stand);
# traegt ein Volume den Stempel des ANDEREN Stands, wird verweigert (zwei Staende nie auf einem Cache).
preflight_caches() {
  local d m st
  for d in "${FLASHINFER_WORKSPACE_BASE:-$HOME}/.cache/flashinfer" "${CUTE_DSL_CACHE_DIR:-/root/.cache/cute-dsl}" \
           "${SGLANG_CACHE_DIR:-$CACHE_DIR}" "${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}" "$HOME/.cache/tvm-ffi" \
           "${TRITON_CACHE_DIR:-$HOME/.triton}"; do
    mkdir -p "$d" 2>/dev/null || true
    m=nein; mountpoint -q "$d" 2>/dev/null && m=ja
    [ "$m" = nein ] && [ "$d" = "${TORCH_EXTENSIONS_DIR:-}" ] && mountpoint -q "$(dirname "$d")" 2>/dev/null && m=ja
    if [ "$m" = nein ]; then
      say "WARN CACHE nicht persistent: $d liegt in der Container-Schicht -- JIT/Autotune (5-29 s je Shape) bei JEDEM Neustart; -v <volume>:$d"
      continue
    fi
    [ -n "${STAND:-}" ] || continue
    st=$(cat "$d/.htsglang-stand" 2>/dev/null || true)
    if [ -n "$st" ] && [ "$st" != "$STAND" ]; then
      refuse CACHE "$d traegt den Stempel von Stand '$st', dieser Lauf ist Stand '$STAND' -- je Stand eigene Cache-Volumes"
    fi
    if [ -z "$st" ]; then FIRST_BOOT_CACHE=1; echo "$STAND" > "$d/.htsglang-stand" 2>/dev/null || true; fi
    say "Cache $d persistent (Stand ${STAND:-?})"
  done
}

run_preflight() {
  ep_cache_link   # fLLiper Schritt 5: der andere Cache-Name zeigt auf das Volume (vor preflight_caches und der Saat)
  preflight_caches
  preflight_gpus
  preflight_shm
  preflight_memory
  preflight_tree
  preflight_model
  preflight_state
  preflight_bar1
}

nccl_barlink_off() {   # barlink KOMPLETT aus: keine Rest-Variablen, die ohne Flags doch barlink bauen liessen
  local v
  for v in SGLANG_BARLINK SGLANG_BARLINK_TRANSPORT SGLANG_BARLINK_PP_TRANSPORT; do
    if [ -n "${!v+x}" ]; then say "nccl: $v='${!v}' aus der Umgebung entfernt"; unset "$v"; fi
  done
}

resolve_transport() {
  : "${HTSGLANG_TRANSPORT:=bar1}"
  local nccl_status=${PROFILE_NCCL_STATUS:-unproven}
  case "$HTSGLANG_TRANSPORT" in
    bar1)
      [ "$BAR1_OK" = "1" ] || refuse TRANSPORT "bar1 verlangt, BAR1-Kette unvollstaendig: ${BAR1_WHY[*]} -- kein Wechsel auf NCCL ohne ausdrueckliches HTSGLANG_TRANSPORT=nccl"
      TRANSPORT=bar1 ;;
    nccl)
      TRANSPORT=nccl
      [ "$nccl_status" = "proven" ] || say "WARN: NCCL ist fuer Profil '$HTSGLANG_PROFILE' UNBELEGT (nie gebootet) -- ausdruecklich verlangt, laeuft" ;;
    auto)
      refuse TRANSPORT "'auto' gibt es nicht (F7: bar1 ist Default, nccl nur ausdruecklich) -- HTSGLANG_TRANSPORT=bar1 oder =nccl" ;;
    *) refuse TRANSPORT "HTSGLANG_TRANSPORT='$HTSGLANG_TRANSPORT' (bar1|nccl)" ;;
  esac
  [ "$TRANSPORT" = "nccl" ] && nccl_barlink_off
  say "TRANSPORT=$TRANSPORT (angefordert $HTSGLANG_TRANSPORT). Beleg nach dem Boot: je Gruppe 'ACHIEVED=bar1' bzw. keine barlink-Zeile"
}

# --- Selbsttest ohne GPU ------------------------------------------------------------
if [ "$SUB" = "selfcheck" ]; then
  preflight_tree
  HTSGLANG_PKG=$PKG "$PY" - <<'EOF'
import os, pathlib
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
# Zwei Staende (25.09.): sglang muss aus dem Baum des gewaehlten Stands kommen, nie aus site-packages.
# fLLiper: alter Baum sglang, umbenannter flliper (HTSGLANG_PKG vom Entrypoint aus dem Baum bestimmt).
import importlib
_pkg = os.environ.get("HTSGLANG_PKG", "sglang")
sglang = importlib.import_module(_pkg)
_tree = os.environ.get("HTSGLANG_TREE", "")
print(_pkg, getattr(sglang, "__version__", "?"), "aus", sglang.__file__)
if _tree and not sglang.__file__.startswith(_tree + "/"):
    print(f"SELFCHECK FEHLER: {_pkg} kommt nicht aus dem Stand-Baum {_tree}")
    raise SystemExit(1)
try:
    print("nccl", torch.cuda.nccl.version())
except Exception as e:  # noqa: BLE001
    print("nccl version unreadable:", e)
import flashinfer
print("flashinfer", flashinfer.__version__)
root = pathlib.Path.home() / ".cache/flashinfer" / flashinfer.__version__
for d in ("120f", "86"):
    ops = sorted(p.name for p in (root / d / "cached_ops").glob("*") if (p / f"{p.name}.so").exists())
    print(f"flashinfer {d}: {len(ops)} vorgebaute Module")
ext = pathlib.Path(os.environ.get("TORCH_EXTENSIONS_DIR", pathlib.Path.home() / ".cache/torch_extensions"))
print("torch_extensions:", sorted(str(p.relative_to(ext)) for p in ext.glob("*/*") if p.is_dir()))
# Abnahme 25.09.: das Image stellte die Erweiterungen unter .../py312_cu130/<name>/ bereit, zur Laufzeit suchte torch
# (TORCH_EXTENSIONS_DIR gesetzt) unter .../<name>/ und baute zwei neu. Darum hier der EFFEKTIVE Suchpfad von torch.
from torch.utils import cpp_extension as _ce
_miss = []
for _so in sorted(ext.glob("**/*.so")):
    _name = _so.parent.name
    if _so.name != f"{_name}.so":
        continue
    # torch bleibt die Autoritaet fuer den Pfad; _get_build_directory legt das Verzeichnis aber an, wenn es fehlt --
    # ein dabei neu entstandenes LEERES Verzeichnis wird wieder entfernt (gemountete Cache-Volumes bleiben sauber).
    _want = pathlib.Path(_ce._get_build_directory(_name, verbose=False)) / f"{_name}.so"
    if not _want.exists():
        _miss.append(f"{_name} (liegt {_so.parent}, torch sucht {_want.parent})")
        try:
            _want.parent.rmdir()
        except OSError:
            pass
print("torch_extensions am effektiven Suchpfad:", "alle" if not _miss else f"FEHLEN {len(_miss)}: " + "; ".join(_miss))
if _miss:
    print("SELFCHECK FEHLER: vorgebaute Erweiterungen liegen nicht dort, wo torch zur Laufzeit sucht -- erster Boot baut neu")
    raise SystemExit(1)
print("tvm-ffi Saat:", len(list((pathlib.Path.home() / ".cache/tvm-ffi").glob("*"))), "Eintraege")
tms = [p.name for p in pathlib.Path(os.environ["SGLANG_WEG2_TMS_OUT_DIR"]).glob("*.so")]
print("tms preload:", tms)
cu13 = pathlib.Path(os.environ.get("SGLANG_WEG2_VENV", "/opt/venv")) / "lib/python3.12/site-packages/nvidia/cu13/lib"
print("cu13 libcudart.so (Rig-Link):", (cu13 / "libcudart.so").exists())
if not tms:
    # Der Launcher baut den Hook bei jedem Boot (build_tms_preload) und verweigert, wenn das scheitert.
    print("SELFCHECK FEHLER: kein TMS-Preload im Image -- jeder weg2-Boot wuerde verweigert (Weg2LaunchRefused)")
    raise SystemExit(1)
nv = pathlib.Path(os.environ.get("SGLANG_BARLINK_BAR1_NV_SOURCE", "/opt/nvidia-open-595"))
print("NV-Header im Image:", (nv / "src/common/sdk/nvidia/inc/nvos.h").is_file())
EOF
  # Referenz ist das Wheel, mit dem DIESES Image gebaut wurde (BUILD_INFO.kernel_wheel_sha256; der Bau prueft es
  # schon in Stufe 19), nicht der im Baum eingefrorene Pin eines aelteren Wheels (rc8duo2: 86;120a-Wheel 605c54e7...
  # gegen Baum-Pin 67f03cfa... = falscher Selfcheck-Fehlschlag). Ohne BUILD_INFO-Eintrag: Baum-Pin wie bisher.
  KW_SHA=$(jq -r '.kernel_wheel_sha256 // empty' "$HOME_DIR/BUILD_INFO.json" 2>/dev/null || true)
  if [ -n "$KW_SHA" ]; then KW_ARG=(--expect-sha256 "$KW_SHA"); else KW_ARG=(--expect-pinned-sha256); fi
  "$PY" -I "$TREE/python/$PKG/srt/utils/kernel_dist_guard.py" \
      --site-packages "$VENV/lib/python3.12/site-packages" --require-arm "${KW_ARG[@]}"
  exit 0
fi

run_preflight
resolve_transport
[ "$SUB" = "preflight" ] && { say "PREFLIGHT fertig -- kein Launch"; exit 0; }
# Abnahme-Befund C (25.09.): T0b lief am Entrypoint vorbei, mit der 7-Arch-Liste des Image-ENV, und baute 3 barlink-
# Varianten (~6 min). Hier laeuft die Probe NACH Umgebungs-Saeuberung und Preflight, also mit der Gruppen-Union wie serve.
if [ "$SUB" = "bar1probe" ]; then
  [ "$TRANSPORT" = "bar1" ] || refuse MODE "bar1probe verlangt HTSGLANG_TRANSPORT=bar1 (ist $TRANSPORT)"
  [ $# -gt 0 ] || set -- 0,1,2 29700
  say "BAR1-PROBE: benchmark/bar1_graph_check.py $*"
  cd "$TREE" && exec "$PY" benchmark/bar1_graph_check.py "$@"
fi

LAUNCH=(-m "$LAUNCHER_MOD" --tree "$TREE" --tag "$HTSGLANG_TAG" --transport "$TRANSPORT" "${PROFILE_ARGS[@]}")
if [ "${HTSGLANG_FORCE:-0}" = 1 ]; then
  LAUNCH+=(--force)
  say "FORCE: Wert-Ablehnungen werden uebergangen (FORCED-PAST im Log), dieser Boot schreibt keine Records; nicht uebergangen: Belegung, fehlendes Modell, Architektur"
fi
# Build-Window-Cap (Operator 25.09., rc8b-Boot dkr27bbar109251735: D starb in der Graph-Capture, das JIT-Kaltbau-Fenster
# stand 160 s offen bei Cap 60 s). Der Launcher SCHREIBT SGLANG_BARLINK_BUILD_WINDOW_CAP_S selbst aus seinem Flag
# --barlink-build-window-cap-s (Default 60, launcher.py:2185/5654) -- eine Env-Variable allein waere wirkungslos. Darum:
#   Profil traegt das Flag schon  -> unveraendert
#   Aufrufer setzt die Env        -> als Flag weitergereicht (Vorrang, sonst ueberschriebe der Launcher sie still)
#   erster Boot auf leerem Cache  -> 900 s (Stempel fehlte in preflight_caches; JIT-Kaltbau mitten in der Graph-Capture)
_bw_in_profile=0; for _a in "${PROFILE_ARGS[@]}"; do [ "$_a" = "--barlink-build-window-cap-s" ] && _bw_in_profile=1; done
grep -q -- '--barlink-build-window-cap-s' "$TREE/python/$PKG/srt/$PDF/launcher.py" 2>/dev/null || _bw_in_profile=2   # Stand kennt das Flag nicht
if [ "$_bw_in_profile" != 0 ]; then :
elif [ -n "$_BWCAP_BY_CALLER" ]; then
  LAUNCH+=(--barlink-build-window-cap-s "$_BWCAP_BY_CALLER")
  say "Build-Window-Cap $_BWCAP_BY_CALLER s (vom Aufrufer, SGLANG_BARLINK_BUILD_WINDOW_CAP_S)"
elif [ "$FIRST_BOOT_CACHE" = 1 ] && [ "$SUB" = serve ]; then
  LAUNCH+=(--barlink-build-window-cap-s 900)
  say "ERSTER BOOT auf leerem Cache-Volume: Build-Window-Cap 900 s statt 60 s (JIT-Kaltbau waehrend der Graph-Capture)"
elif [ "${STAND:-${HTSGLANG_LINE:-}}" = 27b ]; then
  # Operator 25.09. (rc8b INT8 17:35Z und GGUF 17:54Z, beide OHNE neue Kompilate): die Graph-Capture des DFlash-Draft-
  # Workers (decode bs1..6) haelt das Fenster "full cuda-graph capture warmup" im Container 145-160 s (rc8a: 227 s) offen;
  # nach dem 60-s-Cap laeuft das Abbruchwort-Polling wieder -> Bar1CollectiveAborted (all_reduce). 300 s fuer alle 27B-Profile.
  LAUNCH+=(--barlink-build-window-cap-s 300)
  say "Build-Window-Cap 300 s statt Launcher-Default 60 s (27B: Draft-Graph-Capture im Container ~150-230 s, auch warm)"
fi
unset _a _bw_in_profile
# BAR1-Verbindungsfrist beim ERSTEN Boot auf leerem Cache (Operator 25.09., NF-Tod beim ersten Flip im Container: D bot
# seine BAR1-Fenster erst nach ~7 min an -- kalter JIT --, P gab nach 240 s auf, bar1_lanes.py:945 SGLANG_WEG2_BAR1_CONNECT_S,
# fiel auf den Host-Ring). Der Launcher schreibt diese Variable NICHT (anders als das Build-Window-Cap): Env genuegt.
# Beide Staende; nur wenn weder Aufrufer noch Profil sie setzen.
if [ "$FIRST_BOOT_CACHE" = 1 ] && [ "$SUB" = serve ] && [ -z "$_BAR1CONN_BY_CALLER" ] && [ -z "${SGLANG_WEG2_BAR1_CONNECT_S:-}" ]; then
  export SGLANG_WEG2_BAR1_CONNECT_S=900
  say "ERSTER BOOT auf leerem Cache-Volume: SGLANG_WEG2_BAR1_CONNECT_S=900 statt 240 (BAR1-Fenster der Gegengruppe kommen nach kaltem JIT spaet)"
elif [ "$SUB" = serve ] && [ -z "$_BAR1CONN_BY_CALLER" ] && [ -z "${SGLANG_WEG2_BAR1_CONNECT_S:-}" ]; then
  # Warm 480 s (NF 26.09.): die Regel gehoert hierher, nicht ins Profil -- ein Profilwert (gesourct oben) schaltete die
  # Kaltstart-Regel darueber ab, weil sie nur greift, wenn die Variable noch leer ist.
  export SGLANG_WEG2_BAR1_CONNECT_S=480
  say "SGLANG_WEG2_BAR1_CONNECT_S=480 (warmer Cache; kalt 900)"
fi
cd "$TREE"

if [ "$SUB" = "dryrun" ]; then
  say "DRY-RUN: $PY ${LAUNCH[*]} --dry-run $*"
  CUDA_VISIBLE_DEVICES="" exec "$PY" "${LAUNCH[@]}" --dry-run "$@"
fi
[ "$SUB" = "serve" ] || refuse MODE "unbekannter weg2-Untermodus '$SUB'"

STATE=$GPU_ARB/weg2/boot_${HTSGLANG_TAG}.json   # launcher.py:14258-14259
LOGCOPY=$EVIDENCE_DIR/docker_${HTSGLANG_TAG}
mkdir -p "$LOGCOPY"

archive_artifacts() {   # Laufzeit-Artefakte aus GPU_ARB sichern (nur benannte Dateien, nie Schluesseldateien)
  local f
  for f in "$GPU_ARB"/deadman_"${HTSGLANG_TAG}"_*.out "$GPU_ARB"/memts_weg2_"${HTSGLANG_TAG}".csv \
           "$GPU_ARB"/preflight_weg2_"${HTSGLANG_TAG}".log "$STATE" "$GPU_ARB/weg2/boot_${HTSGLANG_TAG}.logpath"; do
    [ -f "$f" ] && cp -p "$f" "$LOGCOPY/" 2>/dev/null || true
  done
}

TORN=0
LPID=""
teardown() {
  [ "$TORN" = "1" ] && return 0; TORN=1
  say "TEARDOWN ($1)"
  [ -n "$LPID" ] && kill -TERM "$LPID" 2>/dev/null && sleep 2
  if declare -F userdash_stop >/dev/null; then userdash_stop; fi   # userdash (USERDASH-DELTA-1001)
  if declare -F rigdash_stop >/dev/null; then rigdash_stop; fi     # rigdash Editor (Auftrag 1995)
  if [ -f "$STATE" ]; then
    "$PY" -m "$LAUNCHER_MOD" --tree "$TREE" --tag "$HTSGLANG_TAG" --teardown "$STATE" \
      >> "$LOGCOPY/teardown.log" 2>&1 || say "teardown rc=$? (siehe $LOGCOPY/teardown.log)"
  fi
  archive_artifacts
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/  nach Teardown: /' >&2 || true
}
# Signale kommen waehrend `wait` sofort an. docker stop -t 180 geben (TERM -> warten -> KILL).
trap 'teardown SIGTERM; exit 0' TERM INT

# userdash (Nutzer-Order 01.10. ~12:55Z, USERDASH-DELTA-1001.md): nur wenn das Image es traegt (/opt/htsglang/userdash,
# Bau mit USERDASH_REV) und HTSGLANG_USERDASH/FLLIPER_USERDASH nicht 0 ist; nur lesend gegen :$FRONT_PORT.
if [ -f /opt/htsglang/userdash/entrypoint_userdash.sh ]; then . /opt/htsglang/userdash/entrypoint_userdash.sh && userdash_start; fi
# rigdash Profil-Editor (Auftrag 1995, Nutzer-Entscheid 05.10.): nur wenn das Image das Paket traegt (Bau mit RIGDASH_REV) und
# HTSGLANG_RIGDASH/FLLIPER_RIGDASH nicht 0 ist; Port 30081, kein Zugriffsschutz (README: -p 127.0.0.1:30081:30081). Der Planer-Baum ist
# der Stand der gestarteten Profilfamilie ($STAND). Scheitert er, bootet der Server trotzdem.
if [ -f /opt/htsglang/rigdash/entrypoint_rigdash.sh ]; then . /opt/htsglang/rigdash/entrypoint_rigdash.sh && rigdash_start; fi
say "LAUNCH: $PY ${LAUNCH[*]} $*"
"$PY" "${LAUNCH[@]}" "$@" > >(tee -a "$LOGCOPY/launcher.log") 2>&1 &
LPID=$!
set +e
wait "$LPID"
rc=$?
set -e
LPID=""
if [ "$rc" != "0" ]; then
  # Eine Verweigerung nach dem ersten Spawn baut der Launcher selbst ab (cli(), #1248).
  say "Launcher rc=$rc -- Boot nicht zustande gekommen (Launcher-Log: $LOGCOPY/launcher.log)"
  archive_artifacts
  exit "$rc"
fi

if [ "$D_ONLY" = 1 ]; then
  say "LAUNCHED (D-only) -- Aufsicht: D-Server :$FRONT_PORT (Liveness /health; KEINE Front, kein /weg2/state)"
  SUPERVISE_RE="launch_server .*--port ${FRONT_PORT}( |\$)"; SUPERVISE_WHAT=D-Server
else
  say "LAUNCHED -- Aufsicht: Front :$FRONT_PORT (Liveness /health, Readiness /weg2/state == serving)"
  # fLLiper (RENAME_PLAN 8.11): die Front heisst im umbenannten Baum flliper.srt.pdflip.front.
  SUPERVISE_RE="(sglang[.]srt[.]weg2|flliper[.]srt[.]pdflip)[.]front .*--port ${FRONT_PORT}"; SUPERVISE_WHAT=Front
fi
while :; do
  if ! pgrep -f "$SUPERVISE_RE" >/dev/null; then
    say "${SUPERVISE_WHAT}-Prozess ist weg -> Teardown"
    teardown "$(printf '%s' "$SUPERVISE_WHAT" | tr '[:upper:]' '[:lower:]')-dead"
    exit 1
  fi
  sleep 5 & wait $!
done
