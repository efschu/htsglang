// Copyright 2025 SGLang Team
// Licensed under the Apache License, Version 2.0
//
// The extension's event handler: a page-action click injects the content
// script and toggles enhancement, and any way a stream can end without the
// user clicking again -- tab closed, navigated away, extension unloaded --
// sends DELETE for the job.
//
// The DELETE is the polite path, not the safety net. The server's #344b
// liveness watchdog reclaims a video_stream whose consumer went quiet, but
// its timeout is a deliberately generous 300 s because a paused player is a
// normal thing. A client that knows the viewer is gone should say so, and
// then the card is free in milliseconds instead of five minutes.

// Chrome runs this as an MV3 service worker, where importScripts is how a
// sibling file is pulled in. Firefox runs it as an MV3 event page, where
// shared.js is already loaded by the manifest's background.scripts list and
// importScripts does not exist -- hence the guard rather than two copies of
// the file.
if (typeof importScripts === "function" && !self.EnhanceShared) {
  importScripts("shared.js");
}

(function () {
  "use strict";

  var shared = self.EnhanceShared;
  // Both browsers, one namespace. Firefox exposes promise-returning
  // `browser.*`; Chrome's MV3 `chrome.*` also returns promises for the APIs
  // used here, so no polyfill is vendored for the sake of two names.
  var api = self.browser || self.chrome;

  // tabId -> {jobId, serverUrl}. In-memory only: a service worker that is
  // evicted loses this, and the watchdog is what covers that case. Persisting
  // it would mean sending DELETE for jobs the server has long since reclaimed.
  var active = {};

  function loadSettings() {
    return api.storage.local.get(shared.DEFAULT_SETTINGS).then(function (stored) {
      return Object.assign({}, shared.DEFAULT_SETTINGS, stored || {});
    });
  }

  function setBadge(tabId, text, title) {
    try {
      api.action.setBadgeText({ tabId: tabId, text: text });
      if (title) {
        api.action.setTitle({ tabId: tabId, title: title });
      }
    } catch (err) {
      /* badges are cosmetic; never let one fail a handoff */
    }
  }

  function cancelJob(record) {
    if (!record || !record.jobId) {
      return Promise.resolve(false);
    }
    var url = shared.buildCancelUrl(record.serverUrl, record.jobId);
    // keepalive so the request survives the tab (and, in Chrome, often the
    // service worker) going away in the same turn.
    return fetch(url, { method: "DELETE", keepalive: true })
      .then(function (response) {
        return response.ok;
      })
      .catch(function () {
        // The server may already have reclaimed it, or be unreachable. Either
        // way there is nothing further this side can do, and the watchdog is
        // the backstop.
        return false;
      });
  }

  function forget(tabId) {
    var record = active[tabId];
    delete active[tabId];
    return record;
  }

  api.action.onClicked.addListener(function (tab) {
    if (!tab || tab.id === undefined) {
      return;
    }
    loadSettings().then(function (settings) {
      return api.scripting
        .executeScript({
          target: { tabId: tab.id },
          files: ["src/shared.js", "src/content.js"],
        })
        .then(function () {
          return api.tabs.sendMessage(tab.id, {
            type: "enhance-toggle",
            settings: settings,
          });
        })
        .then(function (result) {
          if (!result) {
            return;
          }
          if (result.action === "enhanced") {
            active[tab.id] = {
              jobId: result.jobId,
              serverUrl: result.serverUrl,
            };
            setBadge(tab.id, "ON", "Enhancing via " + result.serverUrl);
          } else if (result.action === "restored") {
            cancelJob(forget(tab.id) || result);
            setBadge(tab.id, "", "Enhance video on the fly");
          } else if (result.action === "refused") {
            setBadge(tab.id, "!", result.reason);
            // A refusal is the whole point of the eligibility check, so it is
            // shown rather than logged. Notifications are not requested as a
            // permission; the title text is where the reason lands.
            console.warn("[video-enhance] not eligible:", result.reason);
          }
        })
        .catch(function (err) {
          setBadge(tab.id, "!", String(err && err.message ? err.message : err));
          console.warn("[video-enhance] toggle failed:", err);
        });
    });
  });

  api.tabs.onRemoved.addListener(function (tabId) {
    cancelJob(forget(tabId));
  });

  api.tabs.onUpdated.addListener(function (tabId, changeInfo) {
    // A navigation replaces the document, and with it the <video> element
    // holding the stream. The socket dies but the server would still wait out
    // the liveness window, so the job is cancelled here explicitly.
    if (changeInfo.url && active[tabId]) {
      cancelJob(forget(tabId));
      setBadge(tabId, "", "Enhance video on the fly");
    }
  });

  api.runtime.onMessage.addListener(function (message, sender, sendResponse) {
    // The options page asks the background to probe a server, because the
    // optional host permission is granted to the extension, not to a page.
    if (!message || message.type !== "probe-capabilities") {
      return false;
    }
    fetch(shared.buildCapabilitiesUrl(message.serverUrl, message.query || {}))
      .then(function (response) {
        return response.json().then(function (body) {
          sendResponse({ ok: response.ok, status: response.status, body: body });
        });
      })
      .catch(function (err) {
        sendResponse({ ok: false, status: 0, error: String(err) });
      });
    return true;
  });
})();
