"""#161: der Zensus muss GETEILTE Bytes von EIGENEN trennen koennen.

Am 22.09. habe ich aus EINER Census-Zeile

    [vram-census] pp0tp0-draft after load: model tensors 3.92 GiB =
        {experts 1.32, embed_tokens 1.18, lm_head 1.18, ...}

zweimal eine Zuordnung behauptet und zweimal zurueckgenommen: erst "der
Draft haelt 2,36 GiB zweite Kopie", dann "das gehoert dem Ziel". BEIDES
folgt nicht aus der Zeile. `named_parameters()` zeigt einen per
`lm_head_from_target()` GETEILTEN Tensor mit seinen vollen Bytes -- er
ist dort nicht von einem eigenen zu unterscheiden.

`data_ptr()` entscheidet es: dieselbe Adresse = dasselbe Byte auf der
Karte. Ohne Peer behauptet die Zeile nichts, statt etwas Falsches.
"""

import unittest

import torch
import torch.nn as nn

from flliper.srt.model_executor.vram_family_census import shared_with


def _paar():
    """Ziel und Draft: lm_head GETEILT, embed_tokens EIGEN."""
    target = nn.Module()
    target.lm_head = nn.Linear(64, 512, bias=False)
    target.embed_tokens = nn.Embedding(512, 64)
    draft = nn.Module()
    draft.lm_head = target.lm_head          # derselbe Tensor
    draft.embed_tokens = nn.Embedding(512, 64)   # eine zweite Kopie
    return target, draft


def _named(m):
    return list(m.named_parameters()) + list(m.named_buffers())


class TestSharedVersusOwn(unittest.TestCase):
    def test_shared_tensor_is_detected(self):
        target, draft = _paar()
        fam, byt = shared_with(_named(draft), target)
        self.assertEqual(byt, target.lm_head.weight.numel() * 4)
        self.assertIn("lm_head", fam)

    def test_own_tensor_does_not_count_as_shared(self):
        target, draft = _paar()
        fam, _ = shared_with(_named(draft), target)
        self.assertNotIn("embed_tokens", fam,
                         "eine echte zweite Kopie darf nie als geteilt gelten")

    def test_without_peer_nothing_is_claimed(self):
        """Der Rueckfall: lieber keine Aussage als eine falsche."""
        _target, draft = _paar()
        self.assertEqual(shared_with(_named(draft), None), ({}, 0))

    def test_mutant_all_shared_is_caught(self):
        """Gegenprobe: teilt man BEIDE, muss der Term auf 100 % gehen."""
        target, draft = _paar()
        draft.embed_tokens = target.embed_tokens
        named = _named(draft)
        _fam, byt = shared_with(named, target)
        self.assertEqual(byt, sum(t.numel() * t.element_size() for _n, t in named))

    def test_broken_peer_never_kills(self):
        """Ein Zensus toetet keinen Boot -- auch nicht mit Unsinn als Peer."""
        _target, draft = _paar()
        self.assertEqual(shared_with(_named(draft), object()), ({}, 0))


if __name__ == "__main__":
    unittest.main()


class TestCensusIsCalledWithPeer(unittest.TestCase):
    """Die Lehre des 22.09.: gebaut ist nicht verdrahtet.

    Sieben Instanzen an einem Tag -- #140, #156, #107, #159,
    --rank-auto-reserve-mib, der #150-Melder, und der Vokabular-Share
    selbst. Jedes Mal sah der Code vollstaendig aus. Deshalb prueft
    dieser Test nicht die Funktion, sondern ihren AUFRUFER.
    """

    def test_scheduler_calls_census_with_peer(self):
        import inspect

        from flliper.srt.managers.scheduler import Scheduler

        source = inspect.getsource(Scheduler._maybe_init_draft_kv_producer)
        self.assertIn("log_vram_family_census", source,
                      "der Peer-Zensus wird nicht gerufen")
        self.assertIn("peer=", source,
                      "ohne peer= sagt die Zeile nichts ueber geteilte Bytes")
        self.assertIn("tp_worker.model_runner.model", source,
                      "der Peer muss das ZIEL-Modell sein")

    def test_it_comes_after_both_models(self):
        """Reihenfolge, nicht nur Vorhandensein -- daran ist #92 gestorben."""
        import inspect

        from flliper.srt.managers.scheduler import Scheduler

        q = inspect.getsource(Scheduler._maybe_init_draft_kv_producer)
        self.assertLess(
            q.index("draft_runner.model"), q.index("log_vram_family_census"),
            "der Zensus muss NACH dem Draft-Modell stehen, sonst misst er nichts",
        )

    def test_model_runner_gets_no_peer(self):
        """Gegenprobe: dort waere er immer None -- kein Peer vorhanden.

        Der ModelRunner haelt nur `is_draft_worker` als Flag, keine
        Referenz auf den Ziel-Runner. Ein peer-Argument dort waere die
        achte Instanz der Klasse gewesen.
        """
        import inspect

        from flliper.srt.model_executor import model_runner as mr

        source = inspect.getsource(mr)
        i = source.index("log_vram_family_census(")
        self.assertNotIn("peer=", source[i:i + 400],
                         "der ModelRunner kennt sein Ziel nicht, er darf "
                         "keinen Peer vortaeuschen")


class TestReporterNamesThread162(unittest.TestCase):
    """#162: die Entscheidung messen statt sie aus dem Zensus zu schliessen.

    Bei tie_word_embeddings=False ist `mtp_builds_own_lm_head` genau
    `not head_from_target`. Der Zensus zeigte einen GEBAUTEN lm_head,
    also war `.get()` False -- obwohl der Contextmanager laeuft. Eine
    ContextVar ist THREAD-lokal; der Thread-Name ist deshalb der
    Unterschied zwischen "Schalter kaputt" und "Schalter kommt nicht an".
    """

    def test_decision_formula_depends_only_on_switch(self):
        from flliper.srt.models.qwen3_5_mtp import mtp_builds_own_lm_head

        # unser Checkpoint: tie_word_embeddings = False
        self.assertFalse(mtp_builds_own_lm_head(True, False),
                         "mit Schalter darf NICHTS gebaut werden")
        self.assertTrue(mtp_builds_own_lm_head(False, False),
                        "ohne Schalter wird gebaut -- das ist w134/w136")

    def test_tie_word_embeddings_ist_nicht_deferrable(self):
        from flliper.srt.models.qwen3_5_mtp import mtp_builds_own_lm_head

        self.assertTrue(mtp_builds_own_lm_head(True, True),
                        "bei tie ist der Head die eigene Embedding")

    def test_reporter_names_thread_and_both_inputs(self):
        import inspect

        from flliper.srt.models import qwen3_5_mtp as m

        q = inspect.getsource(m.build_mtp_lm_head)
        self.assertIn("#162 MTP-LM-HEAD", q)
        for field in ("current_thread", "from_target=", "tie_word_embeddings="):
            self.assertIn(field, q, f"{field} fehlt -- dann sagt die Zeile zu wenig")

    def test_reporter_is_before_the_return(self):
        """Sonst meldet er den Deferred-Fall nie."""
        import inspect

        from flliper.srt.models import qwen3_5_mtp as m

        q = inspect.getsource(m.build_mtp_lm_head)
        self.assertLess(q.index("#162 MTP-LM-HEAD"), q.index("Qwen3_5MtpLmHeadDeferred()"))


class TestDerivationFollowsBaseDecision163(unittest.TestCase):
    """#163: Qwen4Exp-MTP umging den Bau-Pfad der Basis.

    `nn.Module.__init__` statt `super().__init__`, lm_head direkt
    alloziert -- damit lief `build_mtp_lm_head` nie und der
    Deferred-Zweig war tot. fnFL2w137 hat es bewiesen: die #162-Zeile
    kam NIE, der Zensus zeigte trotzdem lm_head 1.18 GiB.
    """

    def test_derivation_asks_base_decision(self):
        import inspect

        from flliper.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

        q = inspect.getsource(Qwen4ExpForCausalLMMTP.__init__)
        self.assertIn("mtp_builds_own_lm_head", q)
        self.assertIn("Qwen3_5MtpLmHeadDeferred", q)

    def test_own_prefix_stays(self):
        """NUR die Entscheidung wird uebernommen, NICHT der Bau.

        build_mtp_lm_head nutzt prefix="lm_head", diese Klasse
        "model.shared_head.head". Ein Austausch der ganzen Funktion
        wuerde die Gewichte unter falschem Namen suchen.
        """
        import inspect

        from flliper.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

        q = inspect.getsource(Qwen4ExpForCausalLMMTP.__init__)
        self.assertIn("model.shared_head.head", q)
        self.assertIn("use_attn_tp_group", q)

    def test_names_are_imported(self):
        """Die Falle des Tages: ein Name, der im Modulraum fehlt.

        Mein eigenes publish_expert_map hing heute genau daran, und ein
        `except` haette es als 'failed' verschluckt.
        """
        from flliper.srt.models import qwen4_exp_mtp as m

        for n in ("mtp_builds_own_lm_head", "Qwen3_5MtpLmHeadDeferred",
                  "_LM_HEAD_FROM_TARGET"):
            self.assertTrue(hasattr(m, n), f"{n} fehlt im Modulraum")
