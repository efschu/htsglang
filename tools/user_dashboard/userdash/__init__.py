"""userdash: the slim user dashboard that runs inside the serving image (no VictoriaMetrics, no host paths).

The developer view stays rigdash (tools/rig_dashboard). This package imports nothing from it and nothing
from the server tree: stdlib only, plus nvidia-ml-py when present.
"""

__version__ = "0.1.0"
