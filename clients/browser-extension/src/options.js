// Copyright 2025 SGLang Team
// Licensed under the Apache License, Version 2.0
//
// The settings page. Two jobs beyond storing preferences: it asks for the
// host permission of the configured server (so the manifest can ship with no
// standing host access at all), and it probes GET /v1/video/capabilities so
// the presets shown here are checked against what that deployment reports
// rather than against what this file happens to believe.

(function () {
  "use strict";

  var shared = self.EnhanceShared;
  var api = self.browser || self.chrome;

  var FIELDS = ["serverUrl", "preset", "target", "rifeScale", "startS", "durationS"];

  function el(id) {
    return document.getElementById(id);
  }

  function say(text, tone) {
    var node = el("status");
    node.textContent = text;
    node.style.color = tone === "bad" ? "#a11" : tone === "good" ? "#161" : "#333";
  }

  function fillPresets() {
    var select = el("preset");
    Object.keys(shared.CHAIN_PRESETS).forEach(function (name) {
      var option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    });
    select.addEventListener("change", showPresetHint);
  }

  function showPresetHint() {
    var preset = shared.CHAIN_PRESETS[el("preset").value];
    el("presetHint").textContent = preset ? preset.description : "";
  }

  function load() {
    return api.storage.local.get(shared.DEFAULT_SETTINGS).then(function (stored) {
      var settings = Object.assign({}, shared.DEFAULT_SETTINGS, stored || {});
      FIELDS.forEach(function (key) {
        var value = settings[key];
        el(key).value = value === null || value === undefined ? "" : value;
      });
      showPresetHint();
      return settings;
    });
  }

  function collect() {
    var settings = {};
    FIELDS.forEach(function (key) {
      settings[key] = el(key).value;
    });
    settings.rifeScale = parseFloat(settings.rifeScale) || 1.0;
    settings.startS = parseFloat(settings.startS) || 0;
    settings.durationS = settings.durationS === "" ? null : parseFloat(settings.durationS);
    settings.container = shared.DEFAULT_SETTINGS.container;
    return settings;
  }

  function save() {
    var settings = collect();
    var origin = shared.originPattern(settings.serverUrl);
    if (!origin) {
      say("That server URL cannot be parsed.", "bad");
      return;
    }
    // Requested here, from a user gesture, rather than declared in the
    // manifest: the extension has no idea which server it will be pointed at
    // until somebody types one in, and asking for every host up front to
    // cover that is not a minimal permission.
    api.permissions
      .request({ origins: [origin] })
      .then(function (granted) {
        if (!granted) {
          say("Saved, but access to " + origin + " was declined -- the extension cannot reach that server.", "bad");
        }
        return api.storage.local.set(settings);
      })
      .then(function () {
        say("Saved.", "good");
      })
      .catch(function (err) {
        say("Could not save: " + err, "bad");
      });
  }

  function probe() {
    var settings = collect();
    say("Probing " + settings.serverUrl + " ...");
    api.runtime
      .sendMessage({ type: "probe-capabilities", serverUrl: settings.serverUrl })
      .then(function (result) {
        var report = el("report");
        report.hidden = false;
        if (!result || !result.ok) {
          report.textContent = JSON.stringify(result, null, 2);
          say("The server did not answer with a capability report.", "bad");
          return;
        }
        var body = result.body || {};
        var frontier = body.frontier || {};
        var lines = [
          "tenant: " + body.tenant_id,
          "budget: " + body.budget_mib + " MiB",
          "presets: " + Object.keys(body.chain_presets || {}).join(", "),
          "time range supported: " + body.supports_time_range,
          "measured frontier: " + (frontier.measured ? "yes" : "no -- " + frontier.reason),
        ];
        (frontier.rows || []).forEach(function (row) {
          lines.push(
            "  " +
              row.configuration +
              " " +
              row.resolution +
              " " +
              JSON.stringify(row.options) +
              " -> " +
              row.aggregate_max_fps +
              " fps"
          );
        });
        report.textContent = lines.join("\n");

        var unknown = Object.keys(shared.CHAIN_PRESETS).filter(function (name) {
          return !(body.chain_presets || {})[name];
        });
        if (unknown.length) {
          say("Server reached, but it does not offer preset(s): " + unknown.join(", "), "bad");
        } else {
          say("Server reached; presets match.", "good");
        }
      })
      .catch(function (err) {
        say("Probe failed: " + err, "bad");
      });
  }

  fillPresets();
  load();
  el("save").addEventListener("click", save);
  el("probe").addEventListener("click", probe);
})();
