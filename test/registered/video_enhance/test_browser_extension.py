"""The browser extension's contract with the server and with the browsers.

The extension's own logic is JavaScript and is tested by
``clients/browser-extension/test/run.js`` under node. CI is not assumed to
have a JS runtime, so what runs here is the part that can rot silently across
the language boundary and is checkable from Python:

*   the two manifests are valid, agree with each other, and reference only
    files that exist,
*   the permission set is the minimal one the README promises -- a widened
    permission is a review-worthy change and should fail a test, not slip in,
*   the chain preset names and flags in ``src/shared.js`` are byte-for-byte
    the server's ``CHAIN_PRESETS``, and the endpoint paths it builds are
    routes the server actually mounts,
*   the job ids it generates are ids the server's validator accepts,
*   and every refusal reason the eligibility matrix in the README lists has a
    message in the code.

A rename on either side fails one of these rather than producing requests the
server plans differently than the label promised.
"""

import json
import re
import unittest
from pathlib import Path

from sglang.srt.video_enhance.server import (
    CHAIN_PRESETS,
    JOB_ID_MAX_LENGTH,
    normalize_job_id,
)
from sglang.srt.video_enhance.tenant import TenantConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3] / "clients" / "browser-extension"
SHARED_JS = ROOT / "src" / "shared.js"

#: Exactly what the extension may ask for. activeTab plus scripting is the
#: click-to-inject pattern: no standing access to any page. storage holds the
#: settings. Host access to the configured server is optional and requested at
#: runtime from the options page, because which server that is only becomes
#: known when somebody types one in.
EXPECTED_PERMISSIONS = {"activeTab", "scripting", "storage"}


def _js_source() -> str:
    return SHARED_JS.read_text()


class ManifestTest(CustomTestCase):
    def _manifests(self) -> dict[str, dict]:
        return {
            name: json.loads((ROOT / name).read_text())
            for name in ("manifest.json", "manifest.firefox.json")
        }

    def test_both_manifests_are_manifest_v3(self):
        for name, manifest in self._manifests().items():
            with self.subTest(manifest=name):
                self.assertEqual(manifest["manifest_version"], 3)

    def test_the_two_manifests_describe_the_same_extension(self):
        chrome, firefox = (
            self._manifests()["manifest.json"],
            self._manifests()["manifest.firefox.json"],
        )
        for field in ("name", "version", "description", "permissions"):
            with self.subTest(field=field):
                self.assertEqual(chrome[field], firefox[field])

    def test_the_permission_set_is_the_minimal_one(self):
        for name, manifest in self._manifests().items():
            with self.subTest(manifest=name):
                self.assertEqual(set(manifest["permissions"]), EXPECTED_PERMISSIONS)
                # No standing host access: the server origin is granted at
                # runtime, and no content script is declared, so nothing runs
                # on a page until the user clicks the action.
                self.assertNotIn("host_permissions", manifest)
                self.assertNotIn("content_scripts", manifest)
                self.assertIn("optional_host_permissions", manifest)

    def test_every_referenced_file_exists(self):
        chrome = self._manifests()["manifest.json"]
        firefox = self._manifests()["manifest.firefox.json"]
        referenced = [
            chrome["background"]["service_worker"],
            chrome["options_ui"]["page"],
            *firefox["background"]["scripts"],
        ]
        for relative in referenced:
            with self.subTest(file=relative):
                self.assertTrue((ROOT / relative).is_file(), relative)

    def test_the_files_the_background_injects_exist(self):
        """executeScript names its files as strings; a typo is silent at build."""
        background = (ROOT / "src" / "background.js").read_text()
        match = re.search(r"files:\s*\[(.*?)\]", background, re.S)
        self.assertIsNotNone(match, "no executeScript file list found")
        for relative in re.findall(r'"([^"]+)"', match.group(1)):
            with self.subTest(file=relative):
                self.assertTrue((ROOT / relative).is_file(), relative)

    def test_firefox_loads_shared_js_before_the_background_script(self):
        """Firefox has no importScripts; order in the manifest is the load."""
        scripts = self._manifests()["manifest.firefox.json"]["background"]["scripts"]
        self.assertLess(
            scripts.index("src/shared.js"), scripts.index("src/background.js")
        )

    def test_chrome_pulls_shared_js_in_through_import_scripts(self):
        background = (ROOT / "src" / "background.js").read_text()
        self.assertIn('importScripts("shared.js")', background)
        # ... but only where importScripts exists, or the Firefox event page
        # throws on load.
        self.assertIn('typeof importScripts === "function"', background)


class PresetContractTest(CustomTestCase):
    """The preset table exists twice, in two languages. It must not diverge."""

    def _js_presets(self) -> dict:
        source = _js_source()
        start = source.index("var CHAIN_PRESETS = {")
        start = source.index("{", start)
        depth = 0
        for offset in range(start, len(source)):
            if source[offset] == "{":
                depth += 1
            elif source[offset] == "}":
                depth -= 1
                if depth == 0:
                    literal = source[start : offset + 1]
                    break
        else:  # pragma: no cover - only reachable if the file is truncated
            self.fail("CHAIN_PRESETS literal is not closed")
        # A JS object literal with bare keys and trailing commas is not JSON;
        # quoting the keys and dropping the trailing commas makes it one.
        literal = re.sub(r"(\w+):", r'"\1":', literal)
        literal = re.sub(r",(\s*[}\]])", r"\1", literal)
        return json.loads(literal)

    def test_the_preset_names_match_the_server(self):
        self.assertEqual(sorted(self._js_presets()), sorted(CHAIN_PRESETS))

    def test_every_preset_flag_matches_the_server(self):
        js = self._js_presets()
        for name, expected in CHAIN_PRESETS.items():
            with self.subTest(preset=name):
                self.assertEqual(js[name], expected)

    def test_the_options_page_offers_the_presets_from_shared_js(self):
        """Not a hardcoded list in the HTML, which would be a third copy."""
        options = (ROOT / "src" / "options.js").read_text()
        self.assertIn("shared.CHAIN_PRESETS", options)

    def test_the_presets_cover_upscale_interpolate_and_both(self):
        """Three modes, each one distinguishable from the other two.

        Asserted on the flags rather than on the names, so a preset that was
        renamed still counts and a preset whose flags were quietly changed
        into a duplicate of another does not. A menu entry that produces the
        same request as its neighbour is a menu entry that lies.
        """
        shapes = {
            name: (
                bool(preset.get("enable_sr") or preset.get("enable_resize")),
                int(preset.get("fps_multiplier", 1)) > 1,
            )
            for name, preset in CHAIN_PRESETS.items()
        }
        self.assertEqual(
            sorted(shapes.values()),
            sorted([(True, False), (False, True), (True, True)]),
            f"expected upscale / interpolate / both, got {shapes}",
        )


class FallbackContractTest(CustomTestCase):
    """The engine-unreachable path spans two files and one message name.

    The content script puts the original source back and posts a message; the
    background clears the badge and cancels the job. Nothing in either file
    fails to load if the two names drift apart -- the swap simply stops being
    undone, and the viewer is left with a black player. That is the failure
    this test exists to make loud.
    """

    MESSAGE = "enhance-failed"

    def _read(self, name: str) -> str:
        return (ROOT / "src" / name).read_text()

    def test_the_content_script_reports_a_swap_that_produced_no_frame(self):
        content = self._read("content.js")
        self.assertIn(f'type: "{self.MESSAGE}"', content)
        # The report is worth nothing if the element was not put back first.
        self.assertIn("restore(el)", content)

    def test_the_background_handles_the_report_the_content_script_sends(self):
        background = self._read("background.js")
        self.assertIn(f'message.type === "{self.MESSAGE}"', background)
        self.assertIn("cancelJob(", background)

    def test_the_swap_is_watched_for_both_kinds_of_dead_engine(self):
        """An error event covers a refusal; a timer covers a socket that hangs."""
        content = self._read("content.js")
        self.assertIn('addEventListener("error"', content)
        self.assertIn("FIRST_FRAME_TIMEOUT_MS", content)
        # loadeddata is the success signal that disarms both.
        self.assertIn('addEventListener("loadeddata"', content)

    def test_a_seek_on_a_live_stream_is_answered_rather_than_ignored(self):
        content = self._read("content.js")
        self.assertIn('addEventListener("seeking"', content)
        self.assertIn("seek-unavailable", content)

    def test_the_click_handler_is_reachable_for_an_acceptance_run(self):
        """The toolbar button is browser chrome and cannot be clicked by CDP.

        The acceptance harness calls the function the listener is registered
        with, in the real service worker. If that name stops being published
        the harness stops testing the real path, so it is pinned here.
        """
        background = self._read("background.js")
        self.assertIn("api.action.onClicked.addListener(toggleTab)", background)
        self.assertIn("self.enhanceToggleTab = toggleTab", background)


class EndpointContractTest(CustomTestCase):
    def _constant(self, name: str) -> str:
        match = re.search(rf'var {name} = "([^"]+)"', _js_source())
        self.assertIsNotNone(match, f"{name} not found in shared.js")
        return match.group(1)

    def test_the_paths_shared_js_builds_are_routes_the_server_mounts(self):
        from sglang.srt.video_enhance.server import create_app

        app = create_app(TenantConfig(budget_mib=8192))
        paths = {route.path for route in app.routes}
        self.assertIn(self._constant("ENHANCE_PATH"), paths)
        self.assertIn(self._constant("CAPABILITIES_PATH"), paths)
        # The cancel URL is the enhance path plus the job id segment.
        self.assertIn(self._constant("ENHANCE_PATH") + "/{job_id}", paths)

    def test_the_query_parameter_names_are_the_ones_the_endpoint_takes(self):
        from sglang.srt.video_enhance.server import create_app

        app = create_app(TenantConfig(budget_mib=8192))
        route = next(
            r
            for r in app.routes
            if r.path == "/v1/video/enhance" and "GET" in (r.methods or ())
        )
        accepted = {param.name for param in route.dependant.query_params}
        built = set(re.findall(r'\["(\w+)",', _js_source()))
        # Only the pairs the URL builder pushes, which all live in that array
        # literal form. Anything it sends that the endpoint does not accept is
        # a parameter FastAPI would silently ignore.
        unknown = built - accepted
        self.assertEqual(
            unknown, set(), f"shared.js sends unknown parameters: {unknown}"
        )


class JobIdContractTest(CustomTestCase):
    def test_the_generated_alphabet_is_one_the_server_accepts(self):
        match = re.search(r'var alphabet = "([^"]+)"', _js_source())
        self.assertIsNotNone(match)
        alphabet = match.group(1)
        prefix = re.search(r'var out = "([^"]*)"', _js_source()).group(1)
        # Reconstruct the worst case: the prefix plus every character the
        # generator can emit, checked through the server's own validator.
        candidate = prefix + alphabet
        self.assertLessEqual(len(candidate), JOB_ID_MAX_LENGTH)
        self.assertEqual(normalize_job_id(candidate), candidate)

    def test_the_generated_length_is_within_the_server_limit(self):
        match = re.search(r"for \(var i = 0; i < (\d+); i\+\+\)", _js_source())
        self.assertIsNotNone(match)
        prefix = re.search(r'var out = "([^"]*)"', _js_source()).group(1)
        self.assertLessEqual(len(prefix) + int(match.group(1)), JOB_ID_MAX_LENGTH)


class EligibilityMatrixTest(CustomTestCase):
    """The README's matrix and the code's refusals must be the same set."""

    REFUSAL_CODES = {
        "drm",
        "mse",
        "stream_object",
        "data_uri",
        "local_file",
        "no_source",
        "no_video",
        "unsupported_scheme",
    }

    def test_every_refusal_code_has_a_message(self):
        source = _js_source()
        block = source[
            source.index("var REFUSALS = {") : source.index("var CAVEATS = {")
        ]
        for code in self.REFUSAL_CODES:
            with self.subTest(code=code):
                self.assertIn(code + ":", block)

    def test_the_readme_documents_every_refusal_code(self):
        readme = (ROOT / "README.md").read_text()
        for code in self.REFUSAL_CODES:
            with self.subTest(code=code):
                self.assertIn("`" + code + "`", readme)

    def test_the_readme_names_the_manual_only_steps(self):
        """Manual coverage has to be labelled as such, not implied."""
        readme = (ROOT / "README.md").read_text()
        for phrase in ("Manual", "node test/run.js"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, readme)


class TestRunnerTest(CustomTestCase):
    def test_the_js_tests_need_no_package_manager(self):
        """`node test/run.js` and nothing else -- asserted, since CI cannot run it."""
        self.assertFalse((ROOT / "package.json").exists())
        self.assertFalse((ROOT / "node_modules").exists())
        runner = (ROOT / "test" / "run.js").read_text()
        requires = set(re.findall(r'require\("([^"]+)"\)', runner))
        self.assertEqual(requires, {"./cases.js"})

    def test_the_case_list_only_requires_the_shared_module(self):
        cases = (ROOT / "test" / "cases.js").read_text()
        requires = set(re.findall(r'require\("([^"]+)"\)', cases))
        self.assertEqual(requires, {"../src/shared.js"})


if __name__ == "__main__":
    unittest.main()
