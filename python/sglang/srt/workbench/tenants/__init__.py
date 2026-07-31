# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The registered idle-work tenants (DESIGN #347 W4).

Each module here implements :class:`~sglang.srt.workbench.tenant.
IdleWorkTenant` for one kind of work. Adding a kind means adding a module and
a name to ``--workbench-tenants``; it does not mean touching the scheduler.
"""
