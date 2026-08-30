// Copyright 2025 SGLang Team
// Licensed under the Apache License, Version 2.0
//
// Injected into the active tab on a page-action click. Reads the page's
// <video> elements, hands the best one's source URL to the enhance endpoint
// by swapping the element's src, and restores the original on a second click.
//
// The swap is watched, not assumed. An engine that is down, refuses the job
// or dies mid-stream leaves a <video> element pointed at a URL that will
// never produce frames, and the page's own player has no idea why -- so this
// file puts the original source back, tells the user in the page, and reports
// the failure to the background. A black player with no explanation is the
// one outcome that is worse than not enhancing at all.
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

  //: How long the enhanced stream has to deliver its first frame before the
  //: swap counts as failed. The endpoint answers with chunked HTTP and starts
  //: producing immediately, so this is a liveness bound, not a quality one: a
  //: server that has not sent a decodable frame in this long is either down,
  //: unreachable through the page's CSP, or wedged. Chrome reports a refused
  //: connection through the error event in milliseconds; this covers the
  //: cases that hang instead of failing.
  var FIRST_FRAME_TIMEOUT_MS = 12000;

  //: Distance from the live position, in seconds, past which a seek is a real
  //: seek rather than the sub-second nudge a media element makes on its own
  //: while a stream starts.
  var SEEK_TOLERANCE_S = 1.5;

  var NOTICE_ID = "__video-enhance-notice";

  /**
   * Say something to the viewer, in the page, over the video.
   *
   * The extension has no notifications permission and a badge tooltip is not
   * a notice -- it is invisible until hovered. This is a positioned overlay
   * with a data attribute, which is also what makes the notice assertable
   * from an automated acceptance run rather than only by eye.
   */
  function notice(el, kind, text) {
    var node = document.getElementById(NOTICE_ID);
    if (!node) {
      node = document.createElement("div");
      node.id = NOTICE_ID;
      node.setAttribute("role", "status");
      node.setAttribute("aria-live", "polite");
      node.style.cssText =
        "position:fixed;z-index:2147483647;max-width:38em;padding:10px 14px;" +
        "font:14px/1.4 system-ui,sans-serif;color:#fff;background:#222;" +
        "border-left:4px solid #e0a020;border-radius:4px;" +
        "box-shadow:0 2px 12px rgba(0,0,0,.4);pointer-events:none;";
      document.body.appendChild(node);
    }
    node.dataset.enhanceNotice = kind;
    node.textContent = text;
    var box = el && el.getBoundingClientRect ? el.getBoundingClientRect() : null;
    // Anchored to the video when there is one, top-left of the viewport when
    // the element has no box (display:none, detached) -- never off-screen.
    node.style.top = (box && box.height ? Math.max(box.top + 12, 8) : 8) + "px";
    node.style.left = (box && box.width ? Math.max(box.left + 12, 8) : 8) + "px";
    clearTimeout(node.__enhanceTimer);
    node.__enhanceTimer = setTimeout(function () {
      if (node.parentNode) {
        node.parentNode.removeChild(node);
      }
    }, 15000);
    return node;
  }

  /** Drop every watcher a swap installed. Idempotent. */
  function unwatch(el) {
    var watch = el.__enhanceWatch;
    if (!watch) {
      return;
    }
    delete el.__enhanceWatch;
    clearTimeout(watch.timer);
    el.removeEventListener("error", watch.onError, true);
    el.removeEventListener("loadeddata", watch.onData);
    el.removeEventListener("timeupdate", watch.onTime);
    el.removeEventListener("seeking", watch.onSeeking);
  }

  //: MediaError codes, spelled out because `el.error.code` is a bare number.
  var MEDIA_ERROR_TEXT = {
    1: "the browser aborted the load",
    2: "a network error reaching the enhance server",
    3: "the enhanced stream could not be decoded",
    4: "the enhance server did not answer, or answered with something that is not a playable stream",
  };

  /**
   * Watch a swapped element and undo the swap if the stream never arrives.
   *
   * Two failure signals, because a dead engine produces either one depending
   * on how it is dead: an `error` event (connection refused, 503, a body that
   * is not decodable) and silence (a socket that accepts and then never
   * writes). The success signal is `loadeddata` -- the first decoded frame,
   * which is exactly the claim "the enhanced stream plays here".
   */
  function watchStream(el, onFailure) {
    unwatch(el);
    var settled = false;

    function fail(reason) {
      if (settled) {
        return;
      }
      settled = true;
      unwatch(el);
      onFailure(reason);
    }

    var watch = {
      onError: function () {
        var code = el.error ? el.error.code : 0;
        fail(MEDIA_ERROR_TEXT[code] || "the enhanced stream failed to load");
      },
      onData: function () {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(watch.timer);
        // The load watchers are done; the seek guard below stays for as long
        // as the element is showing the stream.
        el.removeEventListener("error", watch.onError, true);
        el.removeEventListener("loadeddata", watch.onData);
        // Said once, on the first frame, and not conditionally: whether a
        // given browser fires `seeking` for an unreachable position or
        // silently clamps it is a browser detail, so the limitation is
        // announced rather than left to be discovered by dragging the bar.
        notice(
          el,
          "enhancing",
          "Enhanced stream is playing. It is produced live and has no " +
            "seekable index, so seeking is unavailable -- use Start / " +
            "Duration in the extension options to enhance a different " +
            "stretch. Click the action again to restore the original."
        );
      },
      onTime: function () {
        watch.livePosition = el.currentTime;
      },
      onSeeking: function () {
        // A chunked live body has no byte index to seek into, so the element
        // cannot honour the scrub bar. Saying so beats letting the player
        // stall at a position the stream will never reach.
        var target = el.currentTime;
        if (Math.abs(target - watch.livePosition) <= SEEK_TOLERANCE_S) {
          return;
        }
        try {
          el.currentTime = watch.livePosition;
        } catch (err) {
          /* pre-metadata seeks throw; the notice is the point either way */
        }
        notice(
          el,
          "seek-unavailable",
          "Seeking is not available while enhancing: the enhanced stream is " +
            "produced live and has no seekable index. Use Start / Duration in " +
            "the extension options to enhance a different stretch of the source."
        );
      },
      livePosition: 0,
      timer: setTimeout(function () {
        fail(
          "no frame arrived within " +
            Math.round(FIRST_FRAME_TIMEOUT_MS / 1000) +
            " s"
        );
      }, FIRST_FRAME_TIMEOUT_MS),
    };
    el.__enhanceWatch = watch;
    // Capture phase: a media error is dispatched at the element and does not
    // bubble, and a page listener that stops propagation must not blind us.
    el.addEventListener("error", watch.onError, true);
    el.addEventListener("loadeddata", watch.onData);
    el.addEventListener("timeupdate", watch.onTime);
    el.addEventListener("seeking", watch.onSeeking);
  }

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

  /**
   * Put the original source back.
   *
   * Shared by the second click and by the failure path, because "undo the
   * swap" is the same operation whether the viewer asked for it or the engine
   * forced it. Returns the record so the caller can cancel the job.
   */
  function restore(el) {
    var record = el[ENHANCED_KEY];
    delete el[ENHANCED_KEY];
    unwatch(el);
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
    // From here the swap is provisional. Until a frame arrives the original
    // source is still the thing the viewer is entitled to see, and the record
    // above is what puts it back.
    watchStream(el, function (reason) {
      var undone = restore(el);
      notice(
        el,
        "engine-unreachable",
        "Enhancement is off: " +
          reason +
          " (" +
          settings.serverUrl +
          "). The original video is playing."
      );
      // The background owns the badge and the job registry, and it is the
      // only context with the host permission needed to cancel.
      api.runtime.sendMessage({
        type: "enhance-failed",
        jobId: undone.jobId || jobId,
        serverUrl: settings.serverUrl,
        reason: reason,
      });
    });
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
