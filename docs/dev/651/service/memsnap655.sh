#!/bin/bash
# #655: one TSV line describing host+GTT memory state. Called both from outside
# a load (tag "pre") and from inside boot_ondemand.sh right after the page cache
# drop (tag "postdrop").
tag="${1:-snap}"
d=/sys/class/drm/card1/device
r() { [ -f "$1" ] && cat "$1" || echo 0; }
mi() { awk -v k="$1" '$1==k":"{print $2}' /proc/meminfo; }
# top host RSS consumer that is not the model itself, in kB
topproc=$(ps -eo rss=,comm= --sort=-rss | awk 'NR<=6{printf "%s:%s ", $2, $1}')
gnome=$(ps -eo rss=,args= | grep -iE "gnome-shell|Xorg|gdm|gnome-session" | grep -v grep | awk '{s+=$1} END{print s+0}')
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$(date -u +%H:%M:%S)" "$tag" \
  "$(mi MemFree)" "$(mi MemAvailable)" "$(mi Cached)" "$(mi Shmem)" \
  "$(r $d/mem_info_gtt_used)" "$(r $d/mem_info_vram_used)" \
  "$gnome" "$topproc"
