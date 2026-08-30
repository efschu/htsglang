=== #855 WEDGE SPECIMEN, 2026-08-30 05:05-05:15Z
boot: /spinning/evidence-665-f1/boot_855_gdncov_0840f82601_0830_042735.log
trigger load: club-3090 quality-test.sh --medium --enable-thinking (benchlocal), wedged at scenario IF-12

--- deadman verdict:
DEADMAN[HANG-OR-LIVELOCK] 2026-08-30T05:05:15+00:00 port=30030 health_generate failed 2x (m=25s each) log_age=0s last_line: [2026-08-30 05:05:15] INFO:     127.0.0.1:51448 - "GET /health_generate HTTP/1.1" 503 Service Unavailable

[exited with code 0]
--- log signature (repeating):
PHASE-FLIP armed (tp_to_pp) but NOT QUIESCENT: a chunked prefill is incomplete (strict: the flip would discard it, #856 removed the carry). This rank is holding the flip; it has not announced and is not at the entry.
Health check failed. Server couldn't get a response from detokenizer for last 20 seconds. tic start time: 05:13:33. last_heartbeat time: 05:00:50
--- PP0 main-thread samples (py-spy, 5x 3s apart):
    broadcast_pyobj (utils/common.py:2844)|    _broadcast_reqs_across_ranks (scheduler_components/request_receiver.py:399)|    recv_requests (scheduler_components/request_receiver.py:209)|    wrapper (utils/nvtx_utils.py:109)|
    all_reduce (torch/distributed/distributed_c10d.py:3075)|    wrapper (torch/distributed/c10d_logger.py:83)|    _update_uniform_pool_budget (scheduler.py:6186)|    wrapper (utils/nvtx_utils.py:109)|
    all_reduce (torch/distributed/distributed_c10d.py:3075)|    wrapper (torch/distributed/c10d_logger.py:83)|    _update_uniform_pool_budget (scheduler.py:6186)|    wrapper (utils/nvtx_utils.py:109)|
    all_reduce (torch/distributed/distributed_c10d.py:3075)|    wrapper (torch/distributed/c10d_logger.py:83)|    _update_uniform_pool_budget (scheduler.py:6186)|    wrapper (utils/nvtx_utils.py:109)|
    memory_snapshot (torch/cuda/memory.py:623)|    _takeable_cache_bytes (corridor_admission.py:749)|    spendable_bytes (corridor_admission.py:620)|    granted_width (corridor_admission.py:635)|
