# shellcheck shell=bash
# PLANER-3090 1021: gemeinsame Hilfen der 3090-Profile (27b-3x3090.env, 27b-nvfp4-3x3090.env, 27b-nvfp4-dual-3x3090.env).
# Jede Aenderung gegen das Rig-Profil ist eine REGEL des Profil-Planers (pdflip/propose.py, PLANER-3090 1021) fuer N gleiche Karten,
# keine Messung auf einer 3090 (HOCHRECHNUNG): der erste Boot auf den Karten ist die Messung (FLLIPER_FORCE=1, siehe Kopf der Profile).

# _p3090_set FLAG VALUE: das letzte Vorkommen von FLAG in PROFILE_ARGS bekommt VALUE (argparse: das letzte gewinnt), sonst angehaengt.
_p3090_set() {
  local _f=$1 _v=$2 _k _hit=-1
  for ((_k = 0; _k < ${#PROFILE_ARGS[@]}; _k++)); do [ "${PROFILE_ARGS[_k]}" = "$_f" ] && _hit=$_k; done
  if [ "$_hit" -ge 0 ]; then PROFILE_ARGS[_hit + 1]=$_v; else PROFILE_ARGS+=("$_f" "$_v"); fi
}
# _p3090_drop FLAG: jedes Vorkommen von FLAG samt Wert aus PROFILE_ARGS.
_p3090_drop() {
  local _f=$1 _o=() _k
  for ((_k = 0; _k < ${#PROFILE_ARGS[@]}; _k++)); do
    if [ "${PROFILE_ARGS[_k]}" = "$_f" ]; then _k=$((_k + 1)); continue; fi
    _o+=("${PROFILE_ARGS[_k]}")
  done
  PROFILE_ARGS=("${_o[@]}")
}
# _p3090_apply N: die Planer-Regeln fuer N x RTX 3090 auf PROFILE_ARGS (Herleitung je Zeile im Bericht planer-3090-profile-1021.md).
_p3090_apply() {
  local n=$1 i reserve="" trim="" l15=""
  for ((i = 0; i < n; i++)); do
    reserve+="${reserve:+,}1400"     # --user-reserve-mib: Klassenwert der 3080 (Arch-Zwilling sm86), Rig 1800,1400,1400 (5090,3080,3080)
    trim+="${trim:+,}0"              # D-Extend-Trim: 3080-Klasse 0 (nur die 5090 = TP0 starb am Extend, Rig 1200,0,0)
    if [ "$i" -gt 0 ]; then l15+="${l15:+,}c$i=1792"; fi   # L1.5: c0 = Refill-Rang (Kappe 0, wie Rig); sonst kleinster Halt der 3080-Klasse (Rig c1=7616,c2=1792)
  done
  _p3090_set --user-reserve-mib "$reserve"
  _p3090_set --d-tp-objective maxkv  # gleiche Karten: decode-bs1 = Ratenverhaeltnis 1:1:1 = maxkv-Vektor; ohne W61 (keine Kartenrate der 3090)
  _p3090_set --d-reshard off         # d_reshard ist auf RC9_BASE 58,25,25 (5090:3080:3080) gebaut; gleiche Karten: nichts umzuverteilen
  _p3090_drop --d-reshard-presets
  _p3090_drop --pp-stage-ratio       # der Rig-Schnitt (27B-NVFP4 49,8,7) gilt fuer die 5090; der Schnitt-Solver rechnet ihn fuer die Karten
  _p3090_drop --pp-attn-stage-ratio
  _p3090_set --env-d "FLLIPER_PDFLIP_EXTEND_TRIM_MIB=$trim"
  P3090_L15_MIB=$l15
}

# ---- NCARDS 1021: Setzer fuer die Worte INNERHALB der Gruppen-Strings (--extra-p/--extra-d, --env-p/--env-d), die ``tools/ncards_profile_gen.py``
# aus dem Vorschlag des Profil-Planers (propose, Stufe A) ableitet. Gleiche Regeln wie ``propose.LaunchArgv``: ein Wert wird an Ort und Stelle
# ersetzt, nichts sonst am String angehaengt oder umgebaut; fehlt das Wort, wird es angehaengt. Beide Schreibweisen des Gruppen-Strings
# ("--extra-p TEXT" und "--extra-p=TEXT") werden gelesen und in ihrer Schreibweise zurueckgeschrieben.
# _p3090_gfind FLAG: setzt _GI (Index des Textes in PROFILE_ARGS, letztes Vorkommen, -1 = fehlt) und _GM (sep | eq | none).
_p3090_gfind() {
  local _f=$1 _k
  _GI=-1 _GM=none
  for ((_k = 0; _k < ${#PROFILE_ARGS[@]}; _k++)); do
    if [ "${PROFILE_ARGS[_k]}" = "$_f" ]; then _GI=$((_k + 1)) _GM=sep
    elif [[ "${PROFILE_ARGS[_k]}" == "$_f="* ]]; then _GI=$_k _GM=eq; fi
  done
}
_p3090_gread() { if [ "$_GM" = eq ]; then printf '%s' "${PROFILE_ARGS[_GI]#*=}"; else printf '%s' "${PROFILE_ARGS[_GI]}"; fi; }
_p3090_gwrite() { if [ "$_GM" = eq ]; then PROFILE_ARGS[_GI]="$1=$2"; else PROFILE_ARGS[_GI]=$2; fi; }
# _p3090_extra G FLAG VALUE: das Wort "FLAG VALUE" im --extra-G-String ersetzen (angehaengt, wenn FLAG fehlt).
_p3090_extra() {
  local _g=$1 _f=$2 _v=$3 _t
  _p3090_gfind "--extra-$_g"
  if [ "$_GM" = none ]; then PROFILE_ARGS+=("--extra-$_g" "$_f $_v"); return; fi
  _t=$(_p3090_gread)
  if printf '%s' "$_t" | grep -qE "(^|[[:space:]])${_f}([[:space:]]|=)"; then
    _t=$(printf '%s' "$_t" | sed -E "s#(^|[[:space:]])${_f}([[:space:]]+|=)[^[:space:]]+#\\1${_f} ${_v}#")
  else
    _t="$_t $_f $_v"
  fi
  _p3090_gwrite "--extra-$_g" "$_t"
}
# _p3090_extra_drop G FLAG: das Wort samt Wert aus dem --extra-G-String nehmen.
_p3090_extra_drop() {
  local _g=$1 _f=$2 _t
  _p3090_gfind "--extra-$_g"
  [ "$_GM" != none ] || return 0
  _t=$(_p3090_gread | sed -E "s#(^|[[:space:]])${_f}([[:space:]]+|=)[^[:space:]]+##; s#^[[:space:]]+##")
  _p3090_gwrite "--extra-$_g" "$_t"
}
# _p3090_env G KEY VALUE: "KEY=VALUE" im --env-G-String ersetzen (angehaengt, wenn KEY fehlt).
_p3090_env() {
  local _g=$1 _k=$2 _v=$3 _t
  _p3090_gfind "--env-$_g"
  if [ "$_GM" = none ]; then PROFILE_ARGS+=("--env-$_g" "$_k=$_v"); return; fi
  _t=$(_p3090_gread)
  if printf '%s' "$_t" | grep -qE "(^|;)${_k}="; then
    _t=$(printf '%s' "$_t" | sed -E "s#(^|;)${_k}=[^;]*#\\1${_k}=${_v}#")
  else
    _t="$_t;$_k=$_v"
  fi
  _p3090_gwrite "--env-$_g" "$_t"
}
# _p3090_env_drop G KEY: "KEY=..." aus dem --env-G-String nehmen.
_p3090_env_drop() {
  local _g=$1 _k=$2 _t
  _p3090_gfind "--env-$_g"
  [ "$_GM" != none ] || return 0
  _t=$(_p3090_gread | sed -E "s#(^|;)${_k}=[^;]*##; s#^;##")
  _p3090_gwrite "--env-$_g" "$_t"
}
# _p3090_form KEY VALUE: ein global exportiertes Form-Env des Vorschlags (Launcher kopiert os.environ in Front + beide Gruppen); wirkt, sobald das Profil
# profile_form_env() ueber _p3090_form_apply aufruft (siehe die NF-/27B-Profile). _form kommt vom Entrypoint.
P3090_FORM=()
_p3090_form() { P3090_FORM+=("$1" "$2"); }
_p3090_form_apply() {
  local _k
  for ((_k = 0; _k < ${#P3090_FORM[@]}; _k += 2)); do _form "${P3090_FORM[_k]}" "${P3090_FORM[_k + 1]}"; done
}
