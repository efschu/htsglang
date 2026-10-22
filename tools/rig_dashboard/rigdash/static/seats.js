/* SEATS 1022: tab "Seats & devices" -- the UI of rdx-seatmgr (rdma-display multiseat, seat/rdx_seatmgr.py on the Proxmox host).
   One column per seat container plus "Unassigned".  Every keyboard, mouse, gamepad, USB device and sound card is ONE device
   that belongs to at most one seat; "Belongs to" moves it now (until it is unplugged), "Remember by port / by device" makes
   it a rule in /etc/rdx/seatmgr.json, "Assign a device by key press" gives the next key press on an unassigned device to
   that seat.  A device that was just used (key, button, mouse motion) blinks for 2 s, so one can see which entry is which.
   Computes nothing itself: state and actions go through /api/seats/* (server.py passes them to rdx-seatmgr).  Rig edition
   and LAN only.  RigSeats.render(state, {now}) returns the HTML as text (no DOM, no network) for tests. */
(function (root) {
  "use strict";
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const HOW = {
    "default (monitor)": "default: the seat with a monitor",
    "claim": "taken by key press",
    "manual": "chosen here, until unplugged",
    "port rule": "remembered by port",
    "device rule": "remembered by device",
    "GPU sound": "the seat GPU's own sound",
    "unassigned": "not assigned",
  };
  const ROLE = { keyboard: "keyboard", mouse: "mouse", joystick: "gamepad", touchpad: "touchpad", tablet: "tablet", sound: "sound output" };
  const POLL_MS = 1000, POLL_CLAIM_MS = 400;

  function chips(d) {
    const out = [];
    (d.roles || []).forEach((r) => { if (ROLE[r] && !out.includes(ROLE[r])) out.push(ROLE[r]); });
    if (!out.length) out.push(d.kind === "usb" ? "other USB device" : (d.kind === "input" ? "buttons" : d.kind));
    return out.map((c) => '<span class="st-chip">' + esc(c) + "</span>").join(" ");
  }

  function device(d, seats) {
    const locked = d.assignable === false;
    const opts = ['<option value=""' + (d.seat == null ? " selected" : "") + ">Unassigned</option>"].concat(
      seats.map((s) => '<option value="' + s.seat + '"' + (d.seat === s.seat ? " selected" : "") + ">" + esc(s.name) + "</option>"));
    const hasRule = (d.rules || []).length > 0;
    const nodes = (d.nodes || []).map((n) => esc(n.dev)).join(" ");
    return '<div class="st-dev' + (d.active ? " st-blink" : "") + (locked ? " st-locked" : "") + '" data-id="' + esc(d.id) + '">'
      + '<div class="st-dn"><b>' + esc(d.name || d.id) + "</b> " + chips(d) + "</div>"
      + '<div class="st-how"><span>' + esc(HOW[d.how] || d.how) + "</span>"
      + (d.path ? ' <span class="mono st-path" title="' + esc(d.devid) + '">' + esc(d.path) + "</span>" : "") + "</div>"
      + (locked
        ? '<div class="st-note">Used by the host, never handed to a seat</div>'
        : '<div class="st-ctl"><label><span>Belongs to</span> <select data-act="seat">' + opts.join("") + "</select></label>"
          + '<button type="button" data-act="port"' + (d.seat == null || !d.path ? " disabled" : "") + ">Remember by port</button>"
          + '<button type="button" data-act="device"' + (d.seat == null || !d.devid ? " disabled" : "") + ">Remember by device</button>"
          + (hasRule || (d.how !== "unassigned" && d.how !== "GPU sound") ? '<button type="button" data-act="forget">Forget</button>' : "")
          + "</div>")
      + '<details class="st-nodes"><summary>' + (d.nodes || []).length + ((d.nodes || []).length === 1 ? " device node" : " device nodes") + "</summary>"
      + '<div class="mono">' + nodes + "</div></details></div>";
  }

  function seatHead(s, now) {
    const st = s.running ? '<span class="st-chip st-on">running</span>' : '<span class="st-chip">stopped</span>';
    const mon = s.monitor ? ' <span class="st-chip st-on">monitor connected</span>' : "";
    let claim;
    if (s.claim_s != null) {
      claim = '<div class="st-claim st-wait"><span>Press a key on the device for this seat</span> <span class="num">'
        + Math.ceil(s.claim_s) + ' s</span> <button type="button" data-act="cancel" data-seat="' + s.seat + '">Cancel</button></div>';
    } else {
      const got = s.claimed && now - s.claimed.t < 20
        ? '<div class="st-got"><span>Assigned by key press:</span> <b>' + esc(s.claimed.name || s.claimed.id) + "</b></div>" : "";
      claim = '<div class="st-claim"><button type="button" data-act="claim" data-seat="' + s.seat + '">Assign a device by key press</button></div>' + got;
    }
    return '<div class="st-head"><h3>' + esc(s.name) + "</h3>"
      + '<div class="st-meta"><span class="mono">CT ' + s.ct + "</span> " + st + mon + "</div>" + claim + "</div>";
  }

  function render(state, opt) {
    const now = (opt && opt.now) || Date.now() / 1000;
    if (!state || !Array.isArray(state.seats)) return '<div class="muted">rdx-seatmgr answered nothing</div>';
    const seats = state.seats;
    const cols = seats.map((s) => {
      const devs = (state.devices || []).filter((d) => d.seat === s.seat);
      return '<section class="st-col" data-seat="' + s.seat + '">' + seatHead(s, now)
        + (devs.length ? devs.map((d) => device(d, seats)).join("") : '<div class="muted st-empty">No devices</div>') + "</section>";
    });
    const free = (state.devices || []).filter((d) => d.seat == null);
    cols.push('<section class="st-col st-free"><div class="st-head"><h3>Unassigned</h3>'
      + '<div class="st-meta muted">reaches no seat</div></div>'
      + (free.length ? free.map((d) => device(d, seats)).join("") : '<div class="muted st-empty">No devices</div>') + "</section>");
    return '<div class="st-cols">' + cols.join("") + "</div>";
  }

  // ------------------------------------------------------------------ live part (browser only)
  // ``failed``: the last action was refused -- its line stays until the next action (a state read must not wipe it, review 1022 S1)
  let el = null, shown = false, timer = null, last = "", state = null, busy = false, failed = false;

  function note(msg, bad) {
    const n = document.getElementById("st-msg");
    if (n) { n.textContent = msg || ""; n.className = "st-msg" + (bad ? " st-bad" : ""); }
  }

  async function call(path, body) {
    busy = true;
    try {
      const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const j = await r.json().catch(() => ({}));
      failed = !r.ok;
      if (!r.ok) note(j.error || ("HTTP " + r.status), true); else note("");
    } catch (e) {
      failed = true;
      note(String(e), true);
    } finally {
      busy = false;
      poll();
    }
  }

  function draw() {
    const box = document.getElementById("st-root");
    if (!box || !state) return;
    const a = document.activeElement;
    if (a && a.tagName === "SELECT" && box.contains(a)) return;   // never redraw under an open choice
    const html = render(state);
    if (html === last) return;
    last = html;
    const open = new Set([...box.querySelectorAll("details.st-nodes[open]")].map((d) => d.closest(".st-dev").dataset.id));
    box.innerHTML = html;
    box.querySelectorAll(".st-dev").forEach((d) => { if (open.has(d.dataset.id)) d.querySelector("details").open = true; });
  }

  async function poll() {
    clearTimeout(timer);
    if (!shown) return;
    try {
      const r = await fetch("api/seats/state", { cache: "no-store" });
      const j = await r.json();
      if (!r.ok) { note(j.error || ("HTTP " + r.status), true); }
      else { state = j; if (!busy && !failed) note(""); draw(); }
    } catch (e) {
      note("rdx-seatmgr not reachable: " + e, true);
    }
    const claiming = state && state.seats && state.seats.some((s) => s.claim_s != null);
    if (shown && !document.hidden) timer = setTimeout(poll, claiming ? POLL_CLAIM_MS : POLL_MS);
  }

  function devOf(t) { const d = t.closest(".st-dev"); return d ? d.dataset.id : null; }
  function seatOfDev(id) { const d = state && state.devices.find((x) => x.id === id); return d ? d.seat : null; }

  function mount(container) {
    el = container;
    el.innerHTML = '<div id="st-msg" class="st-msg"></div><div id="st-root"><div class="muted">loading &hellip;</div></div>';
    el.addEventListener("change", (ev) => {
      const t = ev.target;
      if (t.dataset.act !== "seat") return;
      const id = devOf(t);
      call("api/seats/assign", { id, seat: t.value === "" ? null : Number(t.value), remember: null });
      t.blur();
    });
    el.addEventListener("click", (ev) => {
      const t = ev.target.closest("button[data-act]");
      if (!t || t.disabled) return;
      const act = t.dataset.act, id = devOf(t);
      if (act === "port" || act === "device") call("api/seats/assign", { id, seat: seatOfDev(id), remember: act });
      else if (act === "forget") call("api/seats/forget", { id });
      else if (act === "claim") call("api/seats/claim", { seat: Number(t.dataset.seat), seconds: 15 });
      else if (act === "cancel") call("api/seats/claim/cancel", { seat: Number(t.dataset.seat) });
    });
    document.addEventListener("visibilitychange", () => { if (!document.hidden && shown) poll(); });
  }

  function show() {
    if (!el) { const c = document.getElementById("seats-root"); if (!c) return; mount(c); }
    if (!shown) { shown = true; poll(); }
  }
  function hide() { shown = false; clearTimeout(timer); }

  root.RigSeats = { render, show, hide, HOW, ROLE };
})(typeof window !== "undefined" ? window : globalThis);
