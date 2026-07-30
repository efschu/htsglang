"""The BAR1 host steps must not leave a server running. Driven through bash.

WHY THIS EXISTS, from the artifact. Attempt 4 of s11 failed at the graph gate
with ``DMABUF_HOLDER_IOC_HOLD`` ENOMEM -- all seven cases red in 106 s -- and
its smoke was answered by the server of attempt 3, which was still alive on
the host holding all three cards. That looked like a finding about bar1 and was
one about cleanup.

The cause is one missing redirection, and ``host_pids`` of that run proves it,
31 bytes on two lines::

    gestartet, pid 1962637
    1962637

``bar1_boot_start`` returns the pid on stdout and called ``host_run_script``
without redirecting it -- so the boot script's own "gestartet, pid <n>" landed
in the caller's command substitution together with the pid. From there:
``host_dump_and_kill`` asked ``kill -0 <that salad>``, got an error, and took
its early exit "nothing there, so nothing to kill". The server survived every
exit path the step had, including the EXIT trap.

Same family as the r7c finding "load_card_order through a pipe": a function
that returns its value on stdout must not let anything else onto stdout.

Everything here runs under ``bash -u`` against stubbed host helpers. No ssh, no
host, no card -- the leak is a shell property and is falsifiable as one.
"""

import pathlib
import subprocess
import sys
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


_REPO = pathlib.Path(__file__).resolve().parents[4]
_BOOT = _REPO / "scripts" / "gpu_battery" / "_bar1_host_boot.sh"

#: Verbatim what the boot script echoes last, and the reason the pid was dirty.
_BOOT_CHATTER = "gestartet, pid 1962637"

_TREIBER = r"""
set -uo pipefail

: "${STUB_PIDFILE_INHALT=1962637}"     # OHNE Doppelpunkt: eine ausdruecklich
: "${STUB_KILL0_RC:=1}"                 # LEERE Vorgabe soll leer bleiben --
: "${STUB_ALTLAST=}"                    # ":=" haette sie ueberschrieben.
: "${STUB_ALTLAST_REAL:=}"              # gesetzt: die PORT/PROC/VRAM-Zeile
                                         # wirklich lokal ausfuehren statt die
                                         # STUB_ALTLAST-Buchstaben zurueckzugeben.
: "${STUB_KILL0_SEQ:=}"                  # z.B. "0,0,1": erst zweimal "lebt
                                         # noch", dann tot -- stellt den
                                         # verzoegerten Tod nach.

# --- Stubs fuer die Host-Ebene ---------------------------------------------
host_path() { printf '%s\n' "$1"; }

# Genau die Stelle, an der es schiefging: host_run_script reicht die Ausgabe
# des entfernten Skripts durch. Der Stub tut dasselbe.
host_run_script() { echo "@@CHATTER@@"; return 0; }

_STUB_KILL0_IDX=0

host_ssh_for() {
    local budget="$1" cmd="$2"
    case "$cmd" in
        *"kill -0"*)
            if [ -n "$STUB_KILL0_SEQ" ]; then
                local IFS=,
                local -a folge=($STUB_KILL0_SEQ)
                unset IFS
                local letzter=$((${#folge[@]} - 1))
                local idx=$_STUB_KILL0_IDX
                [ "$idx" -gt "$letzter" ] && idx=$letzter
                _STUB_KILL0_IDX=$((_STUB_KILL0_IDX + 1))
                return "${folge[$idx]}"
            fi
            return "$STUB_KILL0_RC" ;;
        cat*)         printf '%s\n' "$STUB_PIDFILE_INHALT"; return 0 ;;
        *PORT=*)
            if [ -n "$STUB_ALTLAST_REAL" ]; then
                # Kein ssh, kein Host, keine Karte -- aber die echte
                # Kommandozeile laeuft, gegen die echte lokale Prozessliste.
                # Das ist die einzige Art, den Selbsttreffer ueberhaupt zu
                # sehen: die kanonische Buchstaben-Variante oben testet nur
                # das Parsen eines fertigen PROC=-Werts, nie das pgrep selbst.
                bash -c "$cmd"
                return 0
            fi
            printf '%s\n' "$STUB_ALTLAST"; return 0 ;;
        *)            return 0 ;;
    esac
}

host_dump_and_kill() {
    printf 'DUMP_AND_KILL[%s]\n' "$1" >> "$PROTOKOLL"
    return 0
}

source "@@BOOT@@"

case "${1:-}" in
  boot_start)
      pid="$(bar1_boot_start /tmp/egal /tmp/pidfile)" || pid="<rc!=0>"
      printf 'PID[%s]\n' "$pid"
      ;;
  pid_ok)
      shift
      if bar1_pid_ok "${1:-}"; then echo "OK"; else echo "NEIN"; fi
      ;;
  kill_server)
      shift
      bar1_kill_host_server "${1:-}" /tmp/pidfile /tmp/dump || echo "RC!=0"
      ;;
  altlast)
      if bar1_altlast_pruefen 30030 /tmp/blocked; then echo "FREI"; else echo "ALTLAST"; fi
      ;;
esac
"""


def _lauf(unterbefehl, *args, **env):
    """Run the driver under `bash -u` with a fresh protocol file."""
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        treiber = d / "treiber.sh"
        treiber.write_text(
            _TREIBER.replace("@@BOOT@@", str(_BOOT))
            .replace("@@CHATTER@@", _BOOT_CHATTER)
        )
        protokoll = d / "protokoll.txt"
        protokoll.write_text("")
        umgebung = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(d),
            "PROTOKOLL": str(protokoll),
            # Auch wenn auf dieser Maschine eine Karte laege: keine wird angefasst.
            "CUDA_VISIBLE_DEVICES": "99",
        }
        umgebung.update({k: str(v) for k, v in env.items()})
        fertig = subprocess.run(
            ["bash", "-u", str(treiber), unterbefehl, *args],
            env=umgebung, capture_output=True, text=True, timeout=60,
        )
        return fertig, protokoll.read_text()


class TestPidOnStdoutOnly(CustomTestCase):
    """The falsifier. Without the `>&2` this test goes red."""

    def test_boot_start_returns_the_bare_pid(self):
        fertig, _ = _lauf("boot_start")
        self.assertIn("PID[1962637]", fertig.stdout, msg=fertig.stdout)

    def test_the_boot_chatter_does_not_reach_stdout(self):
        """It has to go SOMEWHERE -- stderr -- not disappear.

        Silencing the boot message would trade one fault for another: the
        step's log is where a failed boot is read from.
        """
        fertig, _ = _lauf("boot_start")
        self.assertNotIn(_BOOT_CHATTER, fertig.stdout)
        self.assertIn(_BOOT_CHATTER, fertig.stderr)

    def test_a_pidfile_with_noise_still_yields_a_bare_pid(self):
        """The file is filtered too -- belt and braces, the file is the
        second source the cleanup path falls back to."""
        fertig, _ = _lauf("boot_start", STUB_PIDFILE_INHALT="pid: 4242\n")
        self.assertIn("PID[4242]", fertig.stdout, msg=fertig.stdout)

    def test_an_empty_pidfile_is_a_failure_not_an_empty_pid(self):
        fertig, _ = _lauf("boot_start", STUB_PIDFILE_INHALT="")
        self.assertIn("PID[<rc!=0>]", fertig.stdout, msg=fertig.stdout)
        self.assertIn("kein brauchbarer Host-pid", fertig.stderr)


class TestPidValidation(CustomTestCase):
    """`kill -0` cannot tell a malformed pid from a dead one. This can."""

    def test_accepts_a_plain_number(self):
        for gut in ("1", "42", "1962637"):
            fertig, _ = _lauf("pid_ok", gut)
            self.assertEqual(fertig.stdout.strip(), "OK", msg=gut)

    def test_rejects_everything_else(self):
        schlecht = [
            "",
            "0",
            "gestartet, pid 1962637",
            "1962637 1962637",
            "abc",
            "-1",
            "12.5",
            _BOOT_CHATTER + "\n1962637",
        ]
        for wert in schlecht:
            fertig, _ = _lauf("pid_ok", wert)
            self.assertEqual(fertig.stdout.strip(), "NEIN", msg=repr(wert))


class TestCleanupReallyKills(CustomTestCase):
    """Two sources for the pid, and a look afterwards."""

    def test_a_valid_pid_is_killed(self):
        _, protokoll = _lauf("kill_server", "4242")
        self.assertIn("DUMP_AND_KILL[4242]", protokoll)

    def test_a_junk_pid_falls_back_to_the_host_pidfile(self):
        """The regression, exactly.

        With the junk string in the variable the old path did nothing at
        all. Now the pidfile the boot script wrote is the second source --
        which also covers a step that died between boot and assignment.
        """
        _, protokoll = _lauf("kill_server", _BOOT_CHATTER, STUB_PIDFILE_INHALT="777")
        self.assertIn("DUMP_AND_KILL[777]", protokoll)

    def test_an_empty_variable_falls_back_too(self):
        _, protokoll = _lauf("kill_server", "", STUB_PIDFILE_INHALT="888")
        self.assertIn("DUMP_AND_KILL[888]", protokoll)

    def test_nothing_anywhere_kills_nothing(self):
        """Negative control: no pid must not become a kill of something else."""
        _, protokoll = _lauf("kill_server", "", STUB_PIDFILE_INHALT="")
        self.assertEqual(protokoll.strip(), "")

    def test_a_survivor_is_reported_loudly(self):
        """A kill nobody checked is an intention, and the intention was
        already there when the server outlived the run.

        Timeout/poll shrunk to keep this fast -- the bound itself (its
        default 15s/1s) is exercised by TestBoundedKillNachschau below.
        """
        fertig, protokoll = _lauf(
            "kill_server", "4242", STUB_KILL0_RC=0,
            BAR1_KILL_NACHSCHAU_TIMEOUT_S=1, BAR1_KILL_NACHSCHAU_POLL_S=1,
        )
        self.assertIn("DUMP_AND_KILL[4242]", protokoll)
        self.assertIn("lebt nach dem Abraeumen noch", fertig.stderr)
        self.assertIn("RC!=0", fertig.stdout)

    def test_a_dead_process_is_reported_as_cleaned(self):
        fertig, _ = _lauf("kill_server", "4242", STUB_KILL0_RC=1)
        self.assertIn("abgeraeumt", fertig.stdout)
        self.assertNotIn("RC!=0", fertig.stdout)


class TestBoundedKillNachschau(CustomTestCase):
    """The race from 2026-07-30, reproduced through the pid, not the host.

    A single, instant ``kill -0`` right after the kill is blind to the
    ordinary gap between SIGTERM and the process actually leaving the
    process table -- that gap is exactly what let a still-dying (not
    foreign, not stuck -- just not yet reaped) server read as "lebt noch"
    and seed the STOP the next attempt's altlast check hit minutes later.
    """

    def test_a_delayed_death_within_the_bound_still_counts_as_cleaned(self):
        """Kill sent, process dies delayed: the verdict stays a clean
        "abgeraeumt", not a survivor report."""
        fertig, protokoll = _lauf(
            "kill_server", "4242", STUB_KILL0_SEQ="0,0,1",
            BAR1_KILL_NACHSCHAU_TIMEOUT_S=5, BAR1_KILL_NACHSCHAU_POLL_S=1,
        )
        self.assertIn("DUMP_AND_KILL[4242]", protokoll)
        self.assertIn("abgeraeumt", fertig.stdout, msg=fertig.stdout + fertig.stderr)
        self.assertNotIn("RC!=0", fertig.stdout)
        self.assertNotIn("lebt nach dem Abraeumen noch", fertig.stderr)

    def test_a_process_that_never_dies_is_reported_as_its_own_state(self):
        """Not within the bound: an honest "Aufraeumen unvollstaendig" state
        (the existing "lebt noch"/RC!=0 wording) -- and specifically NOT the
        Altlast-STOP wording that would retroactively devalue a run. The
        caller (s11/s12 cleanup()) already treats this as non-fatal
        (`|| true`); this test pins that the message itself never claims
        "Altlast"."""
        fertig, protokoll = _lauf(
            "kill_server", "4242", STUB_KILL0_SEQ="0,0,0,0,0",
            BAR1_KILL_NACHSCHAU_TIMEOUT_S=2, BAR1_KILL_NACHSCHAU_POLL_S=1,
        )
        self.assertIn("DUMP_AND_KILL[4242]", protokoll)
        self.assertIn("lebt nach dem Abraeumen noch", fertig.stderr)
        self.assertNotIn("STOP: Altlast von einem vorherigen Anlauf", fertig.stderr)
        self.assertIn("RC!=0", fertig.stdout)


class TestLeftoverDetection(CustomTestCase):
    """Detect, name, abort -- never kill what might not be ours.

    Three tripwires because each alone is blind: the port says nothing about
    a crashed process still holding cards, the process list says nothing
    about a server under another name, and the VRAM figure does not care who
    holds it -- which is exactly the condition the BAR1 setup depends on.
    """

    SAUBER = "PORT=0\nPROC=0\nVRAM=12, 8, 10,\n"

    def test_a_clean_host_passes(self):
        fertig, _ = _lauf("altlast", STUB_ALTLAST=self.SAUBER)
        self.assertIn("FREI", fertig.stdout, msg=fertig.stdout + fertig.stderr)

    def test_a_busy_port_aborts(self):
        fertig, _ = _lauf("altlast", STUB_ALTLAST="PORT=1\nPROC=0\nVRAM=12, 8, 10,\n")
        self.assertIn("ALTLAST", fertig.stdout)
        self.assertIn("Port-30030-belegt", fertig.stderr)

    def test_a_live_launch_server_aborts(self):
        fertig, _ = _lauf("altlast", STUB_ALTLAST="PORT=0\nPROC=4\nVRAM=12, 8, 10,\n")
        self.assertIn("ALTLAST", fertig.stdout)
        self.assertIn("launch_server-Prozesse=4", fertig.stderr)

    def test_occupied_vram_aborts_and_names_the_card(self):
        """The one that actually fired: the holder returned ENOMEM because
        the cards were still full, and the gate read as a bar1 fault."""
        fertig, _ = _lauf(
            "altlast", STUB_ALTLAST="PORT=0\nPROC=0\nVRAM=12, 19850, 10,\n"
        )
        self.assertIn("ALTLAST", fertig.stdout)
        self.assertIn("GPU1=19850MiB", fertig.stderr)

    def test_the_threshold_is_adjustable_and_respected(self):
        fertig, _ = _lauf(
            "altlast",
            STUB_ALTLAST="PORT=0\nPROC=0\nVRAM=12, 2500, 10,\n",
            BAR1_ALTLAST_MIB=4000,
        )
        self.assertIn("FREI", fertig.stdout, msg=fertig.stdout + fertig.stderr)

    def test_the_check_never_kills(self):
        """It only names what it finds. What runs on those cards need not be
        ours, and a broad pkill is exactly the blast radius the rig rules
        rule out."""
        _, protokoll = _lauf(
            "altlast", STUB_ALTLAST="PORT=1\nPROC=9\nVRAM=20000, 20000, 20000,\n"
        )
        self.assertEqual(protokoll.strip(), "")


class TestStaleBerichtDoesNotSurviveACleanPass(CustomTestCase):
    """The 2026-07-30 finding, exactly: Anlauf 5 wrote 'Altlast:
    launch_server-Prozesse=4' to blocked.txt. Ten minutes later Anlauf 6 ran
    completely clean -- but compose() still read that same, never-cleared
    file and turned a green run into a STOP for a finding that belonged to
    a run already over. The verdict of a completed step has to come from
    THAT step's own artifacts, never a leftover from a different attempt.
    """

    def _bericht_pfad(self):
        return pathlib.Path("/tmp/blocked")

    def test_a_stale_report_does_not_survive_a_clean_pass(self):
        bericht = self._bericht_pfad()
        bericht.write_text("Altlast:launch_server-Prozesse=4\n")
        try:
            fertig, _ = _lauf(
                "altlast",
                STUB_ALTLAST="PORT=0\nPROC=0\nVRAM=12, 8, 10,\n",
            )
            self.assertIn("FREI", fertig.stdout, msg=fertig.stdout + fertig.stderr)
            self.assertFalse(
                bericht.exists(),
                msg="ein sauberer Durchlauf muss den alten Bericht loeschen, "
                    "nicht liegen lassen",
            )
        finally:
            bericht.unlink(missing_ok=True)

    def test_a_failing_pass_overwrites_a_stale_report_with_its_own_finding(self):
        """The file must still do its job for a run that IS blocked -- just
        with THIS run's own finding, not a mix with an older one."""
        bericht = self._bericht_pfad()
        bericht.write_text("Altlast:GPU2=30000MiB\n")
        try:
            fertig, _ = _lauf(
                "altlast",
                STUB_ALTLAST="PORT=0\nPROC=4\nVRAM=12, 8, 10,\n",
            )
            self.assertIn("ALTLAST", fertig.stdout)
            inhalt = bericht.read_text()
            self.assertIn("launch_server-Prozesse=4", inhalt)
            self.assertNotIn("GPU2=30000MiB", inhalt)
        finally:
            bericht.unlink(missing_ok=True)


class TestPgrepSelfMatchTrap(CustomTestCase):
    """The bracket idiom, proven against the real process table.

    Every test above stubs ``host_ssh_for`` to hand back a canned ``PROC=``
    value -- which only ever exercises the parsing, never the ``pgrep``
    line itself. That is exactly the blind spot the real bug lived in: the
    checking shell's own command line carries the search pattern as literal
    text (it has to, to search for it), and unbracketed ``pgrep -f`` counts
    that shell as a match. On the affected host this read "launch_server-
    Prozesse=4" with zero real servers running.

    These tests run the real ``pgrep`` line for real, with
    ``STUB_ALTLAST_REAL`` swapping the ssh hop for a plain local
    ``bash -c`` -- no ssh, no host, no card, same as the rest of this file.
    """

    def _lauf_real(self, **env):
        return _lauf("altlast", STUB_ALTLAST_REAL="1", **env)

    def test_the_checking_shell_does_not_count_itself(self):
        """No decoy anywhere: the checking shell's own command line carries
        the pattern as a string but is not launch_server. PROC must read 0.

        Without the bracket this test is the falsifier -- it goes red on
        its own, on a host running nothing at all.
        """
        fertig, _ = self._lauf_real()
        self.assertIn("FREI", fertig.stdout, msg=fertig.stdout + fertig.stderr)

    def test_a_real_launch_server_named_process_is_still_counted(self):
        """Positive control: an actual process whose command line carries
        the un-bracketed name must still be found -- the fix must not trade
        a false positive for a false negative."""
        attrappe = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)",
             "sglang.launch_server"],
        )
        try:
            fertig, _ = self._lauf_real()
            self.assertIn("ALTLAST", fertig.stdout, msg=fertig.stdout + fertig.stderr)
            self.assertIn("launch_server-Prozesse=", fertig.stderr)
        finally:
            attrappe.terminate()
            attrappe.wait(timeout=5)


class TestBothStepsUseIt(CustomTestCase):
    """s12 boots eight times -- eight chances to leave a server behind."""

    def _text(self, name):
        return (_REPO / "scripts" / "gpu_battery" / name).read_text(encoding="utf-8")

    def test_both_steps_check_for_leftovers_before_booting(self):
        for schritt in ("s11_bar1_e2e.sh", "s12_prefill_kurve.sh"):
            self.assertIn("bar1_altlast_pruefen", self._text(schritt), msg=schritt)

    def test_both_steps_clean_up_through_the_checked_path(self):
        for schritt in ("s11_bar1_e2e.sh", "s12_prefill_kurve.sh"):
            text = self._text(schritt)
            self.assertIn("bar1_kill_host_server", text, msg=schritt)
            self.assertIn("trap cleanup EXIT INT TERM", text, msg=schritt)

    def test_no_step_kills_through_the_unchecked_helper_any_more(self):
        """`host_dump_and_kill` is still the mechanism -- but it is reached
        through `bar1_kill_host_server`, which validates the pid, falls back
        to the pidfile and looks afterwards. A direct call would skip all
        three."""
        for schritt in ("s11_bar1_e2e.sh", "s12_prefill_kurve.sh"):
            zeilen = [
                z for z in self._text(schritt).splitlines()
                if "host_dump_and_kill" in z and not z.lstrip().startswith("#")
            ]
            self.assertEqual(zeilen, [], msg=f"{schritt}: {zeilen}")

    def test_the_boot_wrapper_redirects_the_transport_chatter(self):
        """The one-line root fix, pinned at the source.

        Without it the caller's command substitution collects the boot
        message together with the pid -- which is how host_pids of the real
        run came to hold two lines.
        """
        text = _BOOT.read_text(encoding="utf-8")
        self.assertIn('host_run_script 180 "$script" >&2', text)


if __name__ == "__main__":
    unittest.main()
