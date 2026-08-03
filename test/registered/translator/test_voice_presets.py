# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The 18 preset descriptors, and the render plan that turns them into clips.

The descriptors are data, so these tests are contract tests: the pool sizing
decision, the seed stability that keeps a voice identical across re-renders,
and the refusal to render a language whose sentence is missing.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_voice_presets.py -v
"""

import unittest
from collections import Counter

from sglang.srt.translator.voice_presets import (
    PRESET_DESCRIPTORS,
    RENDER_SENTENCES,
    descriptors_for_class,
    render_plan,
)
from sglang.srt.translator.voices import VoiceClass, VoicePool


class TestPoolShape(unittest.TestCase):
    def test_the_descriptors_are_the_MEASURED_pool_not_the_planned_one(self):
        """18 was the plan; 14 is what survived measurement.

        Four voices were retired on 2026-08-03 because they collided with
        another preset of their own class above the speaker registry's 0.70
        same-speaker line (see `voice_presets._RETIRED` for the numbers).
        Two participants handed a colliding pair are indistinguishable to the
        listener AND to the registry, and DESIGN_466 SS4.3 is explicit that
        distinctness beats class match -- so the pool shape follows the
        measurement, not the plan.

        This test therefore pins the measured shape and deliberately does NOT
        assert `RECOMMENDED_PER_CLASS`: that constant is the sizing rationale
        for how many voices to DESIGN, and conflating it with how many
        survived would make the suite demand a pool we measured to be broken.
        """
        counts = Counter(d.voice_class.value for d in PRESET_DESCRIPTORS)
        self.assertEqual(dict(counts), {"man": 5, "woman": 4, "boy": 3, "girl": 2})
        self.assertEqual(len(PRESET_DESCRIPTORS), 14)

    def test_no_retired_voice_leaked_back_into_the_pool(self):
        from sglang.srt.translator.voice_presets import _RETIRED

        active = {d.voice_id for d in PRESET_DESCRIPTORS}
        retired = {d.voice_id for d in _RETIRED}
        self.assertEqual(active & retired, set())
        # Retired descriptors are KEPT, not deleted: the decision is
        # reversible and the pinned seeds must not be lost.
        self.assertEqual(len(retired), 4)

    def test_every_adult_class_clears_the_thinness_floor(self):
        for voice_class in (VoiceClass.MAN, VoiceClass.WOMAN):
            self.assertGreaterEqual(
                len(descriptors_for_class(voice_class)), VoicePool.MIN_PER_CLASS
            )

    def test_child_voices_together_clear_the_floor(self):
        # boy and girl both serve a CHILD classification, so the floor applies
        # to their union rather than to each separately.
        children = len(descriptors_for_class(VoiceClass.BOY)) + len(
            descriptors_for_class(VoiceClass.GIRL)
        )
        self.assertGreaterEqual(children, VoicePool.MIN_PER_CLASS)


class TestDescriptorHygiene(unittest.TestCase):
    def test_ids_and_seeds_are_unique(self):
        ids = [d.voice_id for d in PRESET_DESCRIPTORS]
        seeds = [d.seed for d in PRESET_DESCRIPTORS]
        self.assertEqual(len(set(ids)), len(ids))
        # A shared seed would make two presets render as the same voice, which
        # defeats the entire point of the pool.
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_descriptions_are_distinct_and_substantial(self):
        descriptions = [d.description for d in PRESET_DESCRIPTORS]
        self.assertEqual(len(set(descriptions)), len(descriptions))
        for description in descriptions:
            self.assertGreater(len(description), 60)

    def test_within_a_class_the_descriptions_differ_on_audible_axes(self):
        # Distinctness is the design goal: within a class, the descriptions
        # must vary on pitch/rate/weight words, not just on flavour.
        axis_words = (
            "low", "high", "deep", "bright", "light", "heavy", "soft",
            "slow", "quick", "brisk", "measured", "thin", "resonant",
        )
        for voice_class in (VoiceClass.MAN, VoiceClass.WOMAN):
            fingerprints = set()
            for descriptor in descriptors_for_class(voice_class):
                text = descriptor.description.lower()
                fingerprints.add(
                    frozenset(w for w in axis_words if w in text)
                )
            self.assertEqual(
                len(fingerprints),
                len(descriptors_for_class(voice_class)),
                f"two {voice_class.value} presets describe the same voice",
            )

    def test_the_directory_layout_matches_what_the_pool_loader_expects(self):
        for descriptor in PRESET_DESCRIPTORS:
            self.assertEqual(
                descriptor.directory_name(), descriptor.voice_class.value
            )
            name = descriptor.filename("de")
            self.assertTrue(name.endswith(".de.wav"))
            # <voice_id>.<language>.wav -- exactly two dot-separated stem parts.
            self.assertEqual(len(name[: -len(".wav")].split(".")), 2)


class TestRenderPlan(unittest.TestCase):
    def test_it_covers_every_preset_in_every_language(self):
        plan = render_plan(["de", "es"], "/tmp/pool")
        self.assertEqual(len(plan), len(PRESET_DESCRIPTORS) * 2)
        paths = [entry["path"] for entry in plan]
        self.assertEqual(len(set(paths)), len(paths))

    def test_each_entry_carries_its_own_seed_and_native_sentence(self):
        plan = render_plan(["de", "es"], "/tmp/pool")
        by_id = {}
        for entry in plan:
            by_id.setdefault(entry["voice_id"], set()).add(entry["seed"])
            self.assertEqual(entry["text"], RENDER_SENTENCES[entry["language"]])
        # One seed per voice, shared across its languages: the same person
        # speaking two languages, not two people.
        for voice_id, seeds in by_id.items():
            self.assertEqual(len(seeds), 1, voice_id)

    def test_a_language_without_a_sentence_is_refused(self):
        # Rendering it from another language's text would bake that language's
        # accent into every preset in the pool.
        with self.assertRaises(ValueError) as ctx:
            render_plan(["de", "zz"], "/tmp/pool")
        self.assertIn("zz", str(ctx.exception))

    def test_the_plan_is_deterministic(self):
        first = render_plan(["de", "es"], "/tmp/pool")
        second = render_plan(["de", "es"], "/tmp/pool")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
