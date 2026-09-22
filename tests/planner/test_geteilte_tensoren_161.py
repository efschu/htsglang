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
