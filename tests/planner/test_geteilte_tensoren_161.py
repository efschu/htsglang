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

from sglang.srt.model_executor.vram_family_census import shared_with


def _paar():
    """Ziel und Draft: lm_head GETEILT, embed_tokens EIGEN."""
    ziel = nn.Module()
    ziel.lm_head = nn.Linear(64, 512, bias=False)
    ziel.embed_tokens = nn.Embedding(512, 64)
    draft = nn.Module()
    draft.lm_head = ziel.lm_head          # derselbe Tensor
    draft.embed_tokens = nn.Embedding(512, 64)   # eine zweite Kopie
    return ziel, draft


def _named(m):
    return list(m.named_parameters()) + list(m.named_buffers())


class TestGeteiltGegenEigen(unittest.TestCase):
    def test_geteilter_tensor_wird_erkannt(self):
        ziel, draft = _paar()
        fam, byt = shared_with(_named(draft), ziel)
        self.assertEqual(byt, ziel.lm_head.weight.numel() * 4)
        self.assertIn("lm_head", fam)

    def test_eigener_tensor_zaehlt_nicht_als_geteilt(self):
        ziel, draft = _paar()
        fam, _ = shared_with(_named(draft), ziel)
        self.assertNotIn("embed_tokens", fam,
                         "eine echte zweite Kopie darf nie als geteilt gelten")

    def test_ohne_peer_wird_nichts_behauptet(self):
        """Der Rueckfall: lieber keine Aussage als eine falsche."""
        _ziel, draft = _paar()
        self.assertEqual(shared_with(_named(draft), None), ({}, 0))

    def test_mutant_alles_geteilt_faellt_auf(self):
        """Gegenprobe: teilt man BEIDE, muss der Term auf 100 % gehen."""
        ziel, draft = _paar()
        draft.embed_tokens = ziel.embed_tokens
        named = _named(draft)
        _fam, byt = shared_with(named, ziel)
        self.assertEqual(byt, sum(t.numel() * t.element_size() for _n, t in named))

    def test_kaputter_peer_toetet_nie(self):
        """Ein Zensus toetet keinen Boot -- auch nicht mit Unsinn als Peer."""
        _ziel, draft = _paar()
        self.assertEqual(shared_with(_named(draft), object()), ({}, 0))


if __name__ == "__main__":
    unittest.main()


class TestDerZensusWirdMitPeerGERUFEN(unittest.TestCase):
    """Die Lehre des 22.09.: gebaut ist nicht verdrahtet.

    Sieben Instanzen an einem Tag -- #140, #156, #107, #159,
    --rank-auto-reserve-mib, der #150-Melder, und der Vokabular-Share
    selbst. Jedes Mal sah der Code vollstaendig aus. Deshalb prueft
    dieser Test nicht die Funktion, sondern ihren AUFRUFER.
    """

    def test_scheduler_ruft_den_zensus_mit_peer(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        quelle = inspect.getsource(Scheduler._maybe_init_draft_kv_producer)
        self.assertIn("log_vram_family_census", quelle,
                      "der Peer-Zensus wird nicht gerufen")
        self.assertIn("peer=", quelle,
                      "ohne peer= sagt die Zeile nichts ueber geteilte Bytes")
        self.assertIn("tp_worker.model_runner.model", quelle,
                      "der Peer muss das ZIEL-Modell sein")

    def test_er_steht_NACH_beiden_modellen(self):
        """Reihenfolge, nicht nur Vorhandensein -- daran ist #92 gestorben."""
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        q = inspect.getsource(Scheduler._maybe_init_draft_kv_producer)
        self.assertLess(
            q.index("draft_runner.model"), q.index("log_vram_family_census"),
            "der Zensus muss NACH dem Draft-Modell stehen, sonst misst er nichts",
        )

    def test_der_modelrunner_bekommt_KEINEN_peer(self):
        """Gegenprobe: dort waere er immer None -- kein Peer vorhanden.

        Der ModelRunner haelt nur `is_draft_worker` als Flag, keine
        Referenz auf den Ziel-Runner. Ein peer-Argument dort waere die
        achte Instanz der Klasse gewesen.
        """
        import inspect

        from sglang.srt.model_executor import model_runner as mr

        quelle = inspect.getsource(mr)
        i = quelle.index("log_vram_family_census(")
        self.assertNotIn("peer=", quelle[i:i + 400],
                         "der ModelRunner kennt sein Ziel nicht, er darf "
                         "keinen Peer vortaeuschen")


class TestDerMelderNenntDenThread162(unittest.TestCase):
    """#162: die Entscheidung messen statt sie aus dem Zensus zu schliessen.

    Bei tie_word_embeddings=False ist `mtp_builds_own_lm_head` genau
    `not head_from_target`. Der Zensus zeigte einen GEBAUTEN lm_head,
    also war `.get()` False -- obwohl der Contextmanager laeuft. Eine
    ContextVar ist THREAD-lokal; der Thread-Name ist deshalb der
    Unterschied zwischen "Schalter kaputt" und "Schalter kommt nicht an".
    """

    def test_die_entscheidungsformel_haengt_nur_am_schalter(self):
        from sglang.srt.models.qwen3_5_mtp import mtp_builds_own_lm_head

        # unser Checkpoint: tie_word_embeddings = False
        self.assertFalse(mtp_builds_own_lm_head(True, False),
                         "mit Schalter darf NICHTS gebaut werden")
        self.assertTrue(mtp_builds_own_lm_head(False, False),
                        "ohne Schalter wird gebaut -- das ist w134/w136")

    def test_tie_word_embeddings_ist_nicht_deferrable(self):
        from sglang.srt.models.qwen3_5_mtp import mtp_builds_own_lm_head

        self.assertTrue(mtp_builds_own_lm_head(True, True),
                        "bei tie ist der Head die eigene Embedding")

    def test_der_melder_nennt_thread_und_beide_eingaben(self):
        import inspect

        from sglang.srt.models import qwen3_5_mtp as m

        q = inspect.getsource(m.build_mtp_lm_head)
        self.assertIn("#162 MTP-LM-HEAD", q)
        for feld in ("current_thread", "from_target=", "tie_word_embeddings="):
            self.assertIn(feld, q, f"{feld} fehlt -- dann sagt die Zeile zu wenig")

    def test_der_melder_steht_VOR_dem_return(self):
        """Sonst meldet er den Deferred-Fall nie."""
        import inspect

        from sglang.srt.models import qwen3_5_mtp as m

        q = inspect.getsource(m.build_mtp_lm_head)
        self.assertLess(q.index("#162 MTP-LM-HEAD"), q.index("Qwen3_5MtpLmHeadDeferred()"))
