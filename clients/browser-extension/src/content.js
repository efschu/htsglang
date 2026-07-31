// Copyright 2025 SGLang Team
// Licensed under the Apache License, Version 2.0
//
// Injected into the active tab on a page-action click. Reads the page's
// <video> elements, hands the best one's source URL to the enhance endpoint
// by swapping the element's src, and restores the original on a second click.
//
// Everything that decides is in shared.js; this file only reads the DOM,
// writes the DOM, and reports back. It is injected with scripting.executeScript
// under activeTab, so it has no standing access to any page.

(function () {
  "use strict";

  var shared = self.EnhanceShared;
  var api = self.browser || self.chrome;

  // Injected fresh on every click, so the listener must not be registered
  // twice; the state it guards is what makes the second click a restore.
  if (self.__enhanceContentLoaded) {
    return;
  }
  self.__enhanceContentLoaded = true;

  // The element currently showing an enhanced stream, and what it showed
  // before. Kept on the element rather than in a map so that a page which
  // removes and recreates its video loses the record with it.
  var ENHANCED_KEY = "__enhanceOriginal";

  /** DOM read: one <video> element as the plain object shared.js decides on. */
  function describeVideo(el) {
    var sourceUrls = [];
    var children = el.querySelectorAll("source");
    for (var i = 0; i < children.length; i++) {
      var value = children[i].getAttribute("src");
      if (value) {
        sourceUrls.push(value);
      }
    }
    var rect = el.getBoundingClientRect();
    return {
      element: el,
      currentSrc: el.currentSrc || "",
      src: el.getAttribute("src") || "",
      sourceUrls: sourceUrls,
      // mediaKeys is the EME handle. Its presence is the only reliable
      // in-page signal that the bytes are decrypted in the CDM and cannot be
      // handed anywhere -- the src often looks perfectly ordinary.
      hasMediaKeys: Boolean(el.mediaKeys),
      hasSrcObject: Boolean(el.srcObject),
      videoWidth: el.videoWidth || 0,
      videoHeight: el.videoHeight || 0,
      clientWidth: Math.round(rect.width),
      clientHeight: Math.round(rect.height),
      playing: !el.paused && !el.ended,
      currentTime: el.currentTime || 0,
      pageUrl: location.href,
    };
  }

  function collect() {
    var out = [];
    var elements = document.querySelectorAll("video");
    for (var i = 0; i < elements.length; i++) {
      out.push(describeVideo(elements[i]));
    }
    return out;
  }

  /** Find an element already showing one of our streams, for the restore path. */
  function findEnhanced(settings) {
    var elements = document.querySelectorAll("video");
    for (var i = 0; i < elements.length; i++) {
      var el = elements[i];
      if (el[ENHANCED_KEY]) {
        return el;
      }
      if (shared.isEnhanceUrl(el.currentSrc || el.src, settings.serverUrl)) {
        return el;
      }
    }
    return null;
  }

  function restore(el) {
    var record = el[ENHANCED_KEY];
    delete el[ENHANCED_KEY];
    if (!record) {
      return { action: "restored", jobId: null };
    }
    el.pause();
    if (record.src) {
      el.setAttribute("src", record.src);
    } else {
      el.removeAttribute("src");
    }
    el.load();
    // Put the viewer back where they were rather than at zero. A restore that
    // silently rewinds is worse than no restore.
    try {
      el.currentTime = record.currentTime || 0;
    } catch (err) {
      /* seeking before metadata is loaded throws; not worth failing over */
    }
    if (record.playing) {
      var resume = el.play();
      if (resume && resume.catch) {
        resume.catch(function () {});
      }
    }
    return { action: "restored", jobId: record.jobId, serverUrl: record.serverUrl };
  }

  function enhance(settings) {
    var descriptors = collect();
    if (descriptors.length === 0) {
      return { action: "refused", reason: shared.REFUSALS.no_video };
    }
    var chosen = shared.pickBest(descriptors);
    var verdict = shared.classify(chosen);
    if (!verdict.eligible) {
      return { action: "refused", reason: verdict.reason, kind: verdict.kind };
    }

    var el = chosen.element;
    var jobId = shared.makeJobId();
    var url;
    try {
      url = shared.buildEnhanceUrl(settings, verdict, jobId);
    } catch (err) {
      return { action: "refused", reason: String(err.message || err) };
    }

    el[ENHANCED_KEY] = {
      src: el.getAttribute("src") || chosen.currentSrc,
      currentTime: chosen.currentTime,
      playing: chosen.playing,
      jobId: jobId,
      serverUrl: settings.serverUrl,
    };
    el.pause();
    // Child <source> elements would win over the src attribute on load(), so
    // the attribute alone is not enough to redirect a page that uses them.
    var children = el.querySelectorAll("source");
    for (var i = 0; i < children.length; i++) {
      children[i].remove();
    }
    el.setAttribute("src", url);
    el.load();
    var started = el.play();
    if (started && started.catch) {
      started.catch(function () {});
    }
    return {
      action: "enhanced",
      jobId: jobId,
      serverUrl: settings.serverUrl,
      sourceUrl: verdict.sourceUrl,
      kind: verdict.kind,
      caveats: verdict.caveats,
      enhanceUrl: url,
    };
  }

  api.runtime.onMessage.addListener(function (message, sender, sendResponse) {
    if (!message || message.type !== "enhance-toggle") {
      return false;
    }
    var settings = message.settings || {};
    var existing = findEnhanced(settings);
    var result = existing ? restore(existing) : enhance(settings);
    sendResponse(result);
    return true;
  });
})();
