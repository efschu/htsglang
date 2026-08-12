# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#539 turnkey autoboot + #604 serving watchdog.

One command brings the whole stack up after a host or container boot, and a
watchdog keeps it up with a defined restart policy. The guiding rule for both
halves is that they never guess: every fact the boot needs is either stated in
the config, measured from the machine, or refused by name (see
:mod:`~sglang.srt.turnkey.refusal`).

Division of labour with systemd, which is the whole architecture in three
lines:

* **systemd owns restart-on-death.** ``Restart=on-failure`` plus its rate
  limiting already do this correctly, including the cgroup bookkeeping.
* **The watchdog owns restart-on-WEDGE**, the failure systemd cannot see: the
  process is alive and answers HTTP 200 while generation hangs (#622 family).
  It detects, then asks systemd to restart the unit.
* **The watchdog never spawns serving.** That is #638's lesson made
  structural: a serving process started BY the watchdog inherits the
  watchdog's cgroup (setsid does not escape a cgroup), so stopping the
  watchdog killed production. A detector that only calls ``systemctl restart``
  cannot have that bug, because the new process is started by systemd into the
  serving unit's own cgroup.
"""

from sglang.srt.turnkey.refusal import Refusal, RefusalError  # noqa: F401

__all__ = ["Refusal", "RefusalError"]
