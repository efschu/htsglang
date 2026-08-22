import shutil, subprocess, sys, tempfile, atexit, signal
from pathlib import Path
R = Path("/spinning/wt-739-class")
C = R/"python/sglang/srt/managers/wedge_class.py"
I = R/"python/sglang/srt/managers/scheduler_components/invariant_checker.py"
S = "test/registered/unit/managers/test_wedge_class_739.py"
M = [
 ("M1 falsifier absorbed into A", C,
  '    if delta == 0 and consumed:\n        return WedgeClass(\n            CLASS_UNCLEAR,',
  '    if delta == 0 and consumed:\n        return WedgeClass(\n            CLASS_PIPELINE_DEAD,',
  ["test_can_fail_no_progress_plus_consumed_post_is_unclear_not_a"]),
 ("M2 mirror falsifier absorbed into B", C,
  '    if delta > 0 and unconsumed:\n        return WedgeClass(\n            CLASS_UNCLEAR,',
  '    if delta > 0 and unconsumed:\n        return WedgeClass(\n            CLASS_POOL_SATURATED,',
  ["test_can_fail_progress_plus_unconsumed_post_is_unclear_not_b"]),
 ("M3 saturation decides the class", C,
  "    if delta == 0 and unconsumed:",
  "    if usage_at_ceiling:\n        return WedgeClass(CLASS_POOL_SATURATED, 'ceiling')\n    if delta == 0 and unconsumed:",
  ["test_can_fail_saturation_alone_never_decides_a_class"]),
 ("M4 call edge cut: no class on the alarm line", I,
  '        detail = f"{detail} | {_wedge_class_for(scheduler, stamp, fwd_now, now)}"',
  '        pass  # mutation: the class never reaches the alarm line',
  ["test_the_shipped_detector_appends_a_class_to_its_alarm",
   "test_the_window_is_stamped_once_and_the_delta_grows"]),
 ("M5 window never cleared", I,
  "        if getattr(scheduler, \"_wedge_class_sample\", None) is not None:\n            scheduler._wedge_class_sample = None",
  "        pass  # mutation: the window survives the clear",
  ["test_the_stamp_clears_when_the_alarm_clears"]),
]
def run():
    p = subprocess.run([sys.executable,"-m","pytest",S,"-q","-p","no:randomly","--no-header","--tb=no"],
        cwd=R, capture_output=True, text=True,
        env={"PYTHONPATH":str(R/"python"),"CUDA_VISIBLE_DEVICES":"","PATH":"/usr/bin:/bin","HOME":"/root"}, timeout=300)
    return {l.split("::")[-1].split(" ")[0].split("[")[0] for l in p.stdout.splitlines() if l.startswith("FAILED")}
for _l,path,old,_n,_e in M:
    if old not in path.read_text():
        print(f"[SETUP FAIL] anchor missing for {_l}"); sys.exit(2)
ok=True; live={}
def restore(*a):
    for p,b in list(live.items()): shutil.copy2(b,p); live.pop(p,None)
atexit.register(restore)
for s in (signal.SIGTERM,signal.SIGINT): signal.signal(s, lambda a,b:(restore(),sys.exit(1)))
for label,path,old,new,expect in M:
    b=tempfile.mktemp(); shutil.copy2(path,b); live[path]=b
    try:
        path.write_text(path.read_text().replace(old,new,1)); red=run()
    finally:
        shutil.copy2(b,path); live.pop(path,None)
    miss=[t for t in expect if t not in red]
    if miss: ok=False
    print(f"[{'OK  ' if not miss else 'FAIL'}] {label}  red={sorted(red) if red else 'NOTHING'}")
    if miss: print(f"        MISSING: {miss}")
print("\nALL MUTANTS KILLED" if ok else "\nSOME SURVIVED")
sys.exit(0 if ok else 1)
