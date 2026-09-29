"""weg2xsn272/289 (task #3): PDFLIP-LOAD-DEVICE sits after the prefetch verdict line."""
import pathlib

from flliper.srt.mem_cache import unified_radix_cache as urc


def test_load_device_emitter_after_prefetch_verdict():
    src = pathlib.Path(urc.__file__).read_text()
    body = src.split("def check_prefetch_progress", 1)[1].split("def _req_id_digest", 1)[0]
    i_verdict = body.index('"HiCache prefetch %s req=%s completed_local=%d')
    i_instr = body.index('"PDFLIP-LOAD-DEVICE req=%s tokens=%d bytes_per_token=%d')
    assert i_verdict < i_instr, "instrument must follow the verdict line"
    assert "operation, \"start_time\"" in body.replace("'", '"')
    assert "get_size_per_token()" in body
    assert "log_prefetched_tokens(loaded_from_storage)" in body[i_instr:]
