"""#1480: an un-backed interior node whose Mamba value the pool evicts is
backed up (and the write joined) BEFORE the device value is freed, under
write_back; other policies and already-backed nodes are untouched."""
import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent


def _comp(policy="write_back", written=1, ongoing=None):
    calls = []
    cache = types.SimpleNamespace(
        cache_controller=types.SimpleNamespace(write_policy=policy),
        root_node=object(),
        ongoing_write_through=ongoing if ongoing is not None else {},
        write_backup=lambda node, write_back=False: calls.append(("write_backup", write_back)) or written,
        writing_check=lambda write_back=False: calls.append(("writing_check", write_back)),
    )
    comp = types.SimpleNamespace(cache=cache, component_type="mamba")
    return comp, calls


def _node(backuped=False, value=object(), host_value=None):
    return types.SimpleNamespace(id=7, backuped=backuped,
                                 component_data={"mamba": types.SimpleNamespace(value=value, host_value=host_value)})


class Test1480(unittest.TestCase):
    def test_unbacked_interior_node_is_backed_up_and_joined(self):
        comp, calls = _comp()
        self.assertTrue(MambaComponent._backup_before_mamba_evict(comp, _node()))
        self.assertEqual(calls, [("write_backup", True), ("writing_check", True)])

    def test_write_through_policy_untouched(self):
        comp, calls = _comp(policy="write_through")
        self.assertFalse(MambaComponent._backup_before_mamba_evict(comp, _node()))
        self.assertEqual(calls, [])

    def test_backed_or_host_present_untouched(self):
        comp, calls = _comp()
        self.assertFalse(MambaComponent._backup_before_mamba_evict(comp, _node(backuped=True)))
        self.assertFalse(MambaComponent._backup_before_mamba_evict(comp, _node(host_value=object())))
        self.assertEqual(calls, [])

    def test_refused_backup_retries_after_draining_inflight(self):
        comp, calls = _comp(written=0, ongoing={1: object()})
        self.assertFalse(MambaComponent._backup_before_mamba_evict(comp, _node()))
        self.assertEqual(calls, [("write_backup", True), ("writing_check", True), ("write_backup", True)])

    def test_failing_backup_is_fail_soft(self):
        comp, calls = _comp()
        def boom(node, write_back=False):
            raise RuntimeError("x")
        comp.cache.write_backup = boom
        self.assertFalse(MambaComponent._backup_before_mamba_evict(comp, _node()))


if __name__ == "__main__":
    unittest.main()
