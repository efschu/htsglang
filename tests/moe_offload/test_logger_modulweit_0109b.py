"""#109b: der #109-Pfad braucht einen MODUL-weiten logger.

fnFL2w61 starb mit `NameError: name 'logger' is not defined` in
`presplit_expert_offload_after_repack` -- nachdem #112/4 den Rang
ueberhaupt erst bis dorthin gebracht hatte. `logger` war nur LOKAL in
einer anderen Funktion gebunden.
"""
import logging

from flliper.srt.layers.moe import expert_offload as eo


def test_logger_ist_modulweit():
    assert isinstance(getattr(eo, "logger", None), logging.Logger)


def test_109_path_finds_it():
    """Der Aufruf steht in einer Funktion OHNE eigene logger-Bindung --
    er loest gegen das Modul auf, oder er wirft am Metall."""
    import inspect

    src = inspect.getsource(eo.presplit_expert_offload_after_repack)
    assert "logger." in src, "the path no longer logs -- test outdated?"
    # keine lokale Bindung in dieser Funktion: dann MUSS das Modul sie haben
    if "logger = " not in src:
        assert isinstance(eo.logger, logging.Logger)
