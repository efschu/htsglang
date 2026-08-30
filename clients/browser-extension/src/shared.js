// Copyright 2025 SGLang Team
// Licensed under the Apache License, Version 2.0
//
// Pure decision logic of the enhance-on-the-fly extension: which videos can
// be handed off, and what URL to hand off to. No DOM, no extension API, no
// network -- everything here is a function of its arguments, which is what
// makes it testable under plain node (see ../test/run.js) while the parts
// that touch a page are only testable by loading the extension in a browser.
//
// Loaded three ways, hence the UMD wrapper: as a classic script injected into
// a page, as a background-script import, and as a CommonJS module in the test
// runner.

(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.EnhanceShared = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // Path of the enhance-by-URL endpoint. A GET that streams fragmented MP4,
  // which is what makes a plain <video> element a client of it.
  var ENHANCE_PATH = "/v1/video/enhance";
  var CAPABILITIES_PATH = "/v1/video/capabilities";

  // These names and their flags must stay identical to CHAIN_PRESETS in
  // python/sglang/srt/video_enhance/server.py. The contract is checked by
  // test/registered/video_enhance/test_browser_extension.py, which reads both
  // files -- a rename on one side fails that test rather than silently
  // producing requests the server plans differently than the label promised.
  var CHAIN_PRESETS = {
    sr_only: {
      enable_sr: true,
      sr_scale: 4,
      enable_resize: true,
      fps_multiplier: 1,
      description:
        "x4 super-resolution and resize to target; source frame rate preserved",
    },
    rife_only: {
      enable_sr: false,
      enable_resize: false,
      fps_multiplier: 2,
      description: "interpolation only; source resolution preserved",
    },
    full_chain: {
      enable_sr: true,
      sr_scale: 4,
      enable_resize: true,
      fps_multiplier: 2,
      description: "x4 super-resolution, resize to target, then interpolation",
    },
  };

  var DEFAULT_SETTINGS = {
    serverUrl: "http://127.0.0.1:8100",
    preset: "rife_only",
    // Only consulted by presets that actually change geometry. rife_only
    // keeps the source size, and sending a different target with it makes the
    // server refuse the request -- so the URL builder drops it there rather
    // than passing a value the preset contradicts.
    target: "3840x2160",
    rifeScale: 1.0,
    container: "video/mp4",
    startS: 0,
    durationS: null,
  };

  // Why a given video cannot be handed off. The extension says which one
  // rather than failing quietly, because "nothing happened" is the single
  // most common way a page-action extension wastes somebody's afternoon.
  var REFUSALS = {
    drm: "This video is DRM-protected (Encrypted Media Extensions). The bytes never leave the browser's content decryption module, so no server can be given the stream.",
    mse: "This video is assembled in the page by Media Source Extensions (a blob: source). There is no URL to hand over -- YouTube, Netflix and most adaptive players are in this category.",
    stream_object:
      "This video plays from a live MediaStream (camera, screen share or WebRTC). There is no fetchable URL.",
    data_uri:
      "This video is embedded as a data: URI. There is nothing for the server to fetch.",
    local_file:
      "This video is a file: URL. The server would have to open that path itself, which only works if the file is on the server's host -- pass it to the endpoint directly in that case.",
    no_source: "This element has no source URL yet.",
    no_video: "No <video> element was found on this page.",
    unsupported_scheme:
      "The source URL uses a scheme the server cannot fetch (only http and https are handed off).",
  };

  // Caveats do not block a handoff; they change what the user should expect.
  var CAVEATS = {
    manifest:
      "This is an HLS/DASH manifest. It works only if the server's ffmpeg can reach the manifest and every segment URL in it.",
    cookies:
      "The server fetches this URL itself, without the browser's cookies or Authorization header. A source behind a login will fail on the server side.",
    unknown_size:
      "The intrinsic size of the video is not known yet; the request uses the element's reported size, which the server will reject if it disagrees with the real source.",
  };

  function isManifestUrl(url) {
    var path = url.split("?")[0].split("#")[0].toLowerCase();
    return /\.(m3u8|mpd)$/.test(path);
  }

  /**
   * Resolve a possibly relative source URL against the page it appeared on.
   * Returns null when it cannot be made absolute, which is treated as "no
   * source" rather than guessed at.
   */
  function absolutize(url, pageUrl) {
    if (!url) {
      return null;
    }
    if (/^[a-z][a-z0-9+.-]*:/i.test(url) || url.indexOf("//") === 0) {
      // Already absolute, or protocol-relative.
      try {
        return new URL(url, pageUrl || undefined).href;
      } catch (err) {
        return null;
      }
    }
    if (!pageUrl) {
      return null;
    }
    try {
      return new URL(url, pageUrl).href;
    } catch (err) {
      return null;
    }
  }

  /**
   * Decide whether one video can be handed to the enhance endpoint.
   *
   * `descriptor` is the plain-object form of a <video> element produced by
   * describeVideo() in content.js: currentSrc, src, sourceUrls (from child
   * <source> tags), hasMediaKeys, hasSrcObject, videoWidth, videoHeight,
   * pageUrl. Splitting the DOM read from the decision is what lets every
   * branch below be exercised without a browser.
   */
  function classify(descriptor) {
    var d = descriptor || {};
    if (d.hasMediaKeys) {
      return refuse("drm");
    }
    if (d.hasSrcObject) {
      return refuse("stream_object");
    }

    // currentSrc is what the element actually loaded; src and <source> are
    // what the page asked for. Preferring currentSrc means an adaptive player
    // that swapped in a blob: is detected as MSE instead of being handed the
    // stale attribute value, which would produce a stream of the wrong thing.
    var candidates = [];
    if (d.currentSrc) {
      candidates.push(d.currentSrc);
    }
    if (d.src && d.src !== d.currentSrc) {
      candidates.push(d.src);
    }
    (d.sourceUrls || []).forEach(function (url) {
      if (candidates.indexOf(url) === -1) {
        candidates.push(url);
      }
    });
    if (candidates.length === 0) {
      return refuse("no_source");
    }

    var first = candidates[0];
    if (first.indexOf("blob:") === 0) {
      return refuse("mse");
    }
    if (first.indexOf("data:") === 0) {
      return refuse("data_uri");
    }

    var absolute = absolutize(first, d.pageUrl);
    if (!absolute) {
      return refuse("no_source");
    }
    if (absolute.indexOf("file:") === 0) {
      return refuse("local_file");
    }
    if (absolute.indexOf("http://") !== 0 && absolute.indexOf("https://") !== 0) {
      return refuse("unsupported_scheme");
    }

    var caveats = [CAVEATS.cookies];
    var kind = "direct";
    if (isManifestUrl(absolute)) {
      kind = "manifest";
      caveats.unshift(CAVEATS.manifest);
    }
    var width = d.videoWidth || 0;
    var height = d.videoHeight || 0;
    if (!width || !height) {
      caveats.push(CAVEATS.unknown_size);
    }
    return {
      eligible: true,
      kind: kind,
      sourceUrl: absolute,
      width: width,
      height: height,
      caveats: caveats,
      reason: "",
    };
  }

  function refuse(code) {
    return {
      eligible: false,
      kind: code,
      sourceUrl: null,
      width: 0,
      height: 0,
      caveats: [],
      reason: REFUSALS[code] || code,
    };
  }

  /**
   * Pick the video worth enhancing when a page has several.
   *
   * Largest visible area wins, with a playing element beating a paused one of
   * the same size. That is a heuristic and is meant to be: the alternative is
   * asking the user to click the video first, which is a worse first
   * interaction than occasionally picking the wrong one of two.
   */
  function pickBest(descriptors) {
    var best = null;
    var bestScore = -1;
    (descriptors || []).forEach(function (d) {
      var area = (d.clientWidth || 0) * (d.clientHeight || 0);
      var score = area * 2 + (d.playing ? 1 : 0);
      if (score > bestScore) {
        bestScore = score;
        best = d;
      }
    });
    return best;
  }

  function trimTrailingSlash(url) {
    return String(url || "").replace(/\/+$/, "");
  }

  /**
   * Is this URL already one of our own enhance streams?
   *
   * Used to make the page action a toggle: a second click restores the
   * original source instead of enhancing the enhanced stream, which would
   * make the server fetch from itself.
   */
  function isEnhanceUrl(url, serverUrl) {
    if (!url) {
      return false;
    }
    return url.indexOf(trimTrailingSlash(serverUrl) + ENHANCE_PATH) === 0;
  }

  /**
   * Build the enhance-by-URL request for one source.
   *
   * `target` is only sent for presets that change geometry. rife_only keeps
   * the source resolution, and the server's chain builder refuses a request
   * whose chain cannot reach the named target -- so passing the settings'
   * target along with rife_only would turn every such request into a 422
   * about a target the preset never promised to hit.
   */
  function buildEnhanceUrl(settings, source, jobId) {
    var s = Object.assign({}, DEFAULT_SETTINGS, settings || {});
    var preset = CHAIN_PRESETS[s.preset];
    if (!preset) {
      throw new Error("unknown chain preset: " + s.preset);
    }
    if (!source || !source.sourceUrl) {
      throw new Error("no source URL to enhance");
    }
    if (!source.width || !source.height) {
      throw new Error(
        "the source resolution is required; the server plans the chain and " +
          "its memory reservation from it"
      );
    }
    var changesGeometry = preset.enable_sr || preset.enable_resize;
    var params = [
      ["source_url", source.sourceUrl],
      ["source_width", source.width],
      ["source_height", source.height],
      ["target", changesGeometry ? s.target : source.width + "x" + source.height],
      ["fps_multiplier", preset.fps_multiplier],
      ["enable_sr", preset.enable_sr ? "true" : "false"],
      ["sr_scale", preset.sr_scale || 4],
      ["rife_scale", s.rifeScale],
      ["container", s.container],
    ];
    if (s.startS) {
      params.push(["start_s", s.startS]);
    }
    if (s.durationS !== null && s.durationS !== undefined && s.durationS !== "") {
      params.push(["duration_s", s.durationS]);
    }
    if (jobId) {
      params.push(["job_id", jobId]);
    }
    var query = params
      .map(function (pair) {
        return encodeURIComponent(pair[0]) + "=" + encodeURIComponent(pair[1]);
      })
      .join("&");
    return trimTrailingSlash(s.serverUrl) + ENHANCE_PATH + "?" + query;
  }

  function buildCancelUrl(serverUrl, jobId) {
    return trimTrailingSlash(serverUrl) + ENHANCE_PATH + "/" + encodeURIComponent(jobId);
  }

  function buildCapabilitiesUrl(serverUrl, query) {
    var q = query || {};
    var parts = Object.keys(q)
      .filter(function (key) {
        return q[key] !== null && q[key] !== undefined && q[key] !== "";
      })
      .map(function (key) {
        return encodeURIComponent(key) + "=" + encodeURIComponent(q[key]);
      });
    var base = trimTrailingSlash(serverUrl) + CAPABILITIES_PATH;
    return parts.length ? base + "?" + parts.join("&") : base;
  }

  /**
   * A job id the extension can name up front.
   *
   * It has to be generated client-side: the URL goes into a <video> element,
   * which never surfaces response headers, so a server-minted id would be
   * unknowable here and DELETE unreachable. The alphabet matches the server's
   * accepted set exactly.
   */
  function makeJobId() {
    var alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
    var out = "ext-";
    for (var i = 0; i < 16; i++) {
      out += alphabet[Math.floor(Math.random() * alphabet.length)];
    }
    return out;
  }

  /** Origin pattern for the optional host permission this server needs. */
  function originPattern(serverUrl) {
    try {
      return new URL(serverUrl).origin + "/*";
    } catch (err) {
      return null;
    }
  }

  return {
    ENHANCE_PATH: ENHANCE_PATH,
    CAPABILITIES_PATH: CAPABILITIES_PATH,
    CHAIN_PRESETS: CHAIN_PRESETS,
    DEFAULT_SETTINGS: DEFAULT_SETTINGS,
    REFUSALS: REFUSALS,
    CAVEATS: CAVEATS,
    absolutize: absolutize,
    buildCancelUrl: buildCancelUrl,
    buildCapabilitiesUrl: buildCapabilitiesUrl,
    buildEnhanceUrl: buildEnhanceUrl,
    classify: classify,
    isEnhanceUrl: isEnhanceUrl,
    isManifestUrl: isManifestUrl,
    makeJobId: makeJobId,
    originPattern: originPattern,
    pickBest: pickBest,
  };
});
