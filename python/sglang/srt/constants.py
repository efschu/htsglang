# GPU Memory Types
GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
GPU_MEMORY_TYPE_WEIGHTS = "weights"
GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"

#: #1273 (Weg-2 weight-byte exchange, spec section 4.1): the NEXTN/MTP draft
#: runner's weights, RESIDENT across the flip and never exchanged.  Group D
#: carries the drafter (1382/1311/1311 MiB measured on boot weg2sb4); group P
#: carries no ``--speculative-*`` in this form, so those bytes have no VRAM
#: source on the other side and the exchange cannot fill them.  Giving them
#: their own tag is what lets "every remaining destination byte has a VRAM
#: source" be TRUE, which is in turn what lets the host ring go to zero
#: instead of to MTP-only.
#:
#: DELIBERATELY ABSENT from ``GPU_MEMORY_ALL_TYPES``: that list is the default
#: pause population (weight_updater.py:1197, :1481).  Adding this tag there
#: would pause the drafter -- the exact opposite of the purpose -- and it is
#: likewise outside the weights family (``is_weights_family_tag``), so it is
#: never in a leg, never in a census and never in a wave.
GPU_MEMORY_TYPE_WEIGHTS_DRAFT = "weights_draft"

GPU_MEMORY_ALL_TYPES = [
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
]

HEALTH_CHECK_RID_PREFIX = "HEALTH_CHECK"

GIB_BYTES = 1073741824  # 1024**3
