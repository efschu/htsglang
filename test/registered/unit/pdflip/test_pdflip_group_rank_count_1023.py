"""27B NVFP4 duo 23.10.: the W7/W10 canonical gate takes the group's rank count
from its argv (tp x pp), not the reference rig's literal 3."""
import inspect

from flliper.srt.pdflip import launcher


def _launcher_text() -> str:
    """The launcher source as one text (launcher.py + launcher_*.py) with the split functions expanded back to the original statement
    order: the launcher was split behaviour-neutrally (refactor 28.10.), see ``_launcher_source.expanded_launcher_text``."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("_launcher_source", pathlib.Path(__file__).with_name("_launcher_source.py"))
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    return ls.expanded_launcher_text()


def test_two_card_groups():
    assert launcher.group_rank_count(["--tp-size", "1", "--pp-size", "2"]) == 2
    assert launcher.group_rank_count(["--tp-size", "2", "--pp-size", "1"]) == 2
    assert launcher.group_rank_count(["--tp-size", "2"]) == 2


def test_reference_rig_unchanged_and_last_wins():
    assert launcher.group_rank_count(["--tp-size", "1", "--pp-size", "3"]) == 3
    assert launcher.group_rank_count(["--tp-size", "3"]) == 3
    assert launcher.group_rank_count(["--tp-size=3", "--pp-size", "1", "--pp-size=2"]) == 6


def test_no_literal_rank_count_at_the_gate():
    src = _launcher_text()
    assert "canonical_state_gate(spec_p, 3," not in src
    assert "canonical_state_gate(spec_d, 3," not in src
