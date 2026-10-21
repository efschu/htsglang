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
