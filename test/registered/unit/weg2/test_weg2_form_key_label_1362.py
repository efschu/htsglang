# SPDX-License-Identifier: Apache-2.0
"""#1362 -- the served-model NAME is a label, and a label may not move the form key.

Fossil (4) of #1362 replaced the hardcoded ``--served-model-name "Qwen3.8-27B"``
with a name that follows the model.  That fix passed its own tests and still
came within one commit of invalidating every ring table on the rig: putting the
flag into the argv moved the form key ``237e36801f4b -> 984effac8fb6``, and a
form key that no longer matches is not a warning -- it is every stored P-side
measurement turning into "not this form", for a MODEL THAT MOVED NO BYTE.

The fix was to list the flag in ``ring_table.FORM_KEY_EXCLUDED_FLAGS``, and it
works.  It was also, until this file, pinned by nothing: the #1362 suite has no
reference to the form key at all.  Anyone dropping the line from the exclusion
list -- or adding the NEXT label flag without listing it -- re-arms the same
silent invalidation, and the failure would not surface at the edit but at the
next boot, as a ring table that mysteriously stopped matching.

So this file pins the PROPERTY, not the formula (the hash values themselves are
free to change with the normalisation):

  (a) the flag is in the exclusion list,
  (b) a LABEL change does not move the key,
  (c) a real FORM change does move it, and
  (d) the identity in the key is the checkpoint, ``--model-path``.

(b) alone would be satisfied by a key that ignores everything, which is why (c)
and (d) are here: together they say the key still discriminates what it is for.
"""

from sglang.srt.weg2 import ring_table
from sglang.test.test_utils import CustomTestCase

#: A minimal group-P argv.  Only flags that are IN the key belong here, so that
#: a single mutation per test is the only difference between two readings.
BASE_ARGV = [
    "--model-path",
    "/models/Qwen3.8-27B-INT4",
    "--tp-size",
    "3",
    "--pp-size",
    "1",
    "--mem-fraction-static",
    "0.9",
]


class TheServedNameIsALabelNotAForm1362(CustomTestCase):
    """``--served-model-name`` names the model to callers; it does not load it."""

    def test_the_flag_is_in_the_exclusion_list(self):
        # The list is the mechanism; (b) below is the behaviour it buys.  Both
        # are pinned, because a future normalisation could keep the behaviour
        # by accident and lose it again on the next edit.
        self.assertIn(
            "--served-model-name",
            ring_table.FORM_KEY_EXCLUDED_FLAGS,
            "--served-model-name is a LABEL: it is the name the front routes on, "
            "not a term of what group P loads. In the key it invalidates every "
            "stored ring table whenever the name's spelling changes.",
        )

    def test_a_label_change_does_not_move_the_key(self):
        bare = ring_table.p_form_key(BASE_ARGV)
        labelled = ring_table.p_form_key(
            BASE_ARGV + ["--served-model-name", "Qwen3.8-27B"]
        )
        renamed = ring_table.p_form_key(
            BASE_ARGV + ["--served-model-name", "some-other-spelling"]
        )
        self.assertEqual(bare[0], labelled[0])
        self.assertEqual(bare[0], renamed[0])
        self.assertEqual(bare[1], labelled[1], "the normalised form string too")

    def test_a_real_form_change_still_moves_the_key(self):
        # Guards against the trivial pass of (b): a key that dropped everything
        # would satisfy the test above and be worthless.
        other_model = ring_table.p_form_key(
            ["--model-path", "/models/Qwen3.5-4B-FP8"] + BASE_ARGV[2:]
        )
        other_shape = ring_table.p_form_key(
            [t if t != "3" else "2" for t in BASE_ARGV]
        )
        bare = ring_table.p_form_key(BASE_ARGV)
        self.assertNotEqual(bare[0], other_model[0], "a different checkpoint is a different form")
        self.assertNotEqual(bare[0], other_shape[0], "a different TP size is a different form")

    def test_the_identity_in_the_key_is_the_model_path(self):
        # What the label stopped carrying, --model-path still carries: the key
        # is allowed to ignore the NAME only because it does not ignore the
        # CHECKPOINT.
        _, form = ring_table.p_form_key(
            BASE_ARGV + ["--served-model-name", "Qwen3.8-27B"]
        )
        self.assertIn("--model-path=/models/Qwen3.8-27B-INT4", form)
        self.assertNotIn("--served-model-name", form)
