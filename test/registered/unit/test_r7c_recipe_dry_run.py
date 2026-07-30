"""The four r7c boot recipes must survive `set -u` from the top to the launch
line, on a machine with no cards.

Why this test exists: round 7c's boot C died at the line that picks the
drafter's card with `CUDA_SMALL: unbound variable`, after the log had already
printed `CUDA_SMALL=1,2` two lines earlier. The call site read
`load_card_order | tee cards.txt`, so the resolver ran in a pipeline subshell:
the echoed lines reached the log while the two assignments died with that
subshell. It cost a boot window to find something that needs no GPU at all --
the recipes therefore carry a dry run (`R7C_DRY_RUN=1`), and this test walks
every one of them through it under `bash -u`.

Two card orders are exercised, not one. With a single fixture the derived
values (boot C's drafter card and its per-rank reserve string) could match by
coincidence; swapping which index is the big card makes them prove they are
actually derived from the resolution.
"""

import pathlib
import re
import subprocess
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_R7C = _REPO_ROOT / "scripts" / "dual_group" / "r7c"
_RECIPES = [
    "boot_a_fp8_reference.sh",
    "boot_b_dense_head.sh",
    "boot_c_dflash_solo_q8.sh",
    "boot_d_lane_reseed.sh",
]

# Verbatim shape of what resolve_cards() prints on the rig, including the two
# header blocks -- the parsing has to survive them, not just the last two lines.
_CARDS_TEMPLATE = """CUDA order (what the flags mean):
  cuda:0  NVIDIA GeForce RTX {n0}
  cuda:1  NVIDIA GeForce RTX {n1}
  cuda:2  NVIDIA GeForce RTX {n2}
NVML order (what nvidia-smi prints):
  0, NVIDIA GeForce RTX 3080, 20480 MiB
  1, NVIDIA GeForce RTX 5090, 32607 MiB
  2, NVIDIA GeForce RTX 3080, 20480 MiB
CUDA_BIG={big}
CUDA_SMALL={small}
"""

# (label, big index, small list, name row) -- the second is the first with the
# big card moved, which is exactly the shift the runbook says not to assume away.
_CARD_ORDERS = [
    ("big_first", "0", "1,2", ("5090", "3080", "3080")),
    ("big_middle", "1", "0,2", ("3080", "5090", "3080")),
]


def _dry_run(recipe: str, big: str, small: str, names) -> subprocess.CompletedProcess:
    """Run one recipe through its dry-run path in a throwaway directory."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        cards = tmpdir / "cards_fixture.txt"
        cards.write_text(
            _CARDS_TEMPLATE.format(
                big=big, small=small, n0=names[0], n1=names[1], n2=names[2]
            )
        )
        (tmpdir / "wt").mkdir()
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmpdir),
            "R7C_DRY_RUN": "1",
            "R7C_CARDS_FILE": str(cards),
            # No card may be touched even if one exists on the test machine.
            "CUDA_VISIBLE_DEVICES": "99",
            "WT": str(tmpdir / "wt"),
            "VENV": str(tmpdir / "venv"),
            "MODEL_ROOT": str(tmpdir / "models"),
            "REPO_ROOT": str(_REPO_ROOT),
            "OUT": str(tmpdir / "out"),
            "LOG": str(tmpdir / "server.log"),
        }
        return subprocess.run(
            ["bash", "-u", str(_R7C / recipe)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )


class TestR7cRecipeDryRun(CustomTestCase):
    def test_no_unbound_variable_in_any_recipe(self):
        for recipe in _RECIPES:
            for label, big, small, names in _CARD_ORDERS:
                with self.subTest(recipe=recipe, cards=label):
                    proc = _dry_run(recipe, big, small, names)
                    combined = proc.stdout + proc.stderr
                    self.assertNotIn(
                        "unbound variable",
                        combined.lower(),
                        msg=(
                            f"{recipe} hit `set -u` in its dry run "
                            f"({label}). Output:\n{combined}"
                        ),
                    )
                    self.assertEqual(
                        proc.returncode,
                        0,
                        msg=f"{recipe} dry run failed ({label}). Output:\n{combined}",
                    )
                    self.assertIn(
                        "DRY RUN LAUNCH:",
                        proc.stdout,
                        msg=(
                            f"{recipe} never reached the launch line, so the "
                            f"flag assembly went unchecked. Output:\n{combined}"
                        ),
                    )

    def test_boot_c_derives_drafter_card_from_the_resolution(self):
        """Boot C's two derived values, against both card orders.

        These are what the crash was about: the drafter goes on the FIRST small
        card, and the raised reserve follows it rather than a fixed index.
        """
        expected = {
            # big=0, small=1,2 -> drafter on 1; reserve 3000 on the big card,
            # RESERVE_HOST on the drafter's, 2700 on the remaining 3080.
            "big_first": ("cuda:1", "3000,5000,2700"),
            # big=1, small=0,2 -> drafter on 0.
            "big_middle": ("cuda:0", "5000,3000,2700"),
        }
        for label, big, small, names in _CARD_ORDERS:
            with self.subTest(cards=label):
                proc = _dry_run("boot_c_dflash_solo_q8.sh", big, small, names)
                self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
                card, reserve = expected[label]
                self.assertIn(f"Drafter-Karte: {card}", proc.stdout)
                self.assertIn(f"--rank-auto-reserve-mib {reserve}", proc.stdout)
                self.assertIn(
                    f"--speculative-draft-gpu {card[len('cuda:') :]}", proc.stdout
                )

    def test_boot_c_passes_the_drafter_as_a_gguf_file(self):
        """The drafter path must be the .gguf FILE, never its directory.

        A GGUF target resolves load_format=gguf, the draft worker inherits it,
        and GGUFModelLoader._prepare_weights (model_loader/loader.py) is
        `os.path.isfile(path) or ValueError`. The directory form cost a boot
        window: the server crashed 2.5 minutes into the target load with
        "qwen3.6-27b-dflash-gguf is not a file."
        """
        for label, big, small, names in _CARD_ORDERS:
            with self.subTest(cards=label):
                proc = _dry_run("boot_c_dflash_solo_q8.sh", big, small, names)
                self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
                launch = [
                    line
                    for line in proc.stdout.splitlines()
                    if line.startswith("DRY RUN LAUNCH:")
                ]
                self.assertEqual(len(launch), 1, msg=proc.stdout)
                flags = launch[0].split()
                self.assertIn("--speculative-draft-model-path", flags)
                draft = flags[flags.index("--speculative-draft-model-path") + 1]
                self.assertTrue(
                    draft.endswith(".gguf"),
                    msg=(
                        "--speculative-draft-model-path must name the .gguf "
                        f"file the GGUF loader opens, got {draft!r}"
                    ),
                )
                # The target is a GGUF file too -- that is what makes the draft
                # worker inherit load_format=gguf in the first place.
                self.assertIn("--model-path", flags)
                self.assertTrue(
                    flags[flags.index("--model-path") + 1].endswith(".gguf")
                )

    def test_wait_for_server_watches_the_server_process(self):
        """A dead server must end the wait, not run out the budget.

        Boot C's crash landed at 04:46 and was collected at 05:14: the readiness
        loop polled a port belonging to a process that no longer existed for 28
        minutes. Checked on the source because the loop needs a real server to
        exercise.
        """
        common = (_R7C / "common.sh").read_text()
        body = common.split("wait_for_server()", 1)
        self.assertEqual(len(body), 2, msg="wait_for_server not found in common.sh")
        self.assertIn(
            "_pid_alive",
            body[1].split("\n}\n", 1)[0],
            msg=(
                "wait_for_server polls only the port -- a crashed server is "
                "waited out to the full budget."
            ),
        )
        self.assertIn(
            "LAUNCH_PIDFILE",
            common,
            msg="launch_server must record its pid file for wait_for_server",
        )

    def test_no_recipe_pipes_load_card_order(self):
        """The source form of the bug, kept out by a ratchet.

        `load_card_order` assigns CUDA_BIG/CUDA_SMALL in its caller. Any pipe
        after it puts the function in a subshell and throws both away, which is
        invisible in the log because the echoed lines still arrive. The function
        takes the log file as an argument instead.
        """
        offenders = []
        for path in sorted(_R7C.glob("*.sh")):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                match = re.search(r"load_card_order\b(.*)$", line)
                if match is None:
                    continue
                # `||` is the fail-fast idiom, not a pipeline; only a single
                # bar redirects the function into a subshell.
                if "|" in match.group(1).replace("||", ""):
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
        self.assertFalse(
            offenders,
            msg=(
                "load_card_order must not be piped -- the assignments would run "
                "in a pipeline subshell and read as unbound at the first use. "
                "Pass the log file as an argument:\n  " + "\n  ".join(offenders)
            ),
        )


if __name__ == "__main__":
    unittest.main()
