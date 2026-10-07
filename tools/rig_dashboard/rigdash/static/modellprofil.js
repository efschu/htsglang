/* Modellprofil schätzen (PROFIL-EDITOR S3, Auftrag 960): das kleine Modul hinter dem Knopf "Modellprofil erstellen".
   Rechnet NICHTS selbst: POST /api/modellprofil/schaetzen liefert das Modellprofil flliper.model/1 (der Schätzer liest nur
   config.json und die Kopfzeilen der Shards, nie ein Gewicht).  Dieses Modul holt es ab und macht daraus Zeilen
   {gruppe, label, wert, src, hinweis} -- die Oberfläche (Auftrag 930) zeichnet sie; tabelle() ist ein fertiger HTML-Baustein.
   Jeder Wert trägt seine Quelle: config | Index | geschätzt | stat.  Nur im Rig-Dashboard (Edition rig). */
(function (root) {
  "use strict";
  const QUELLE = {
    "config": "steht in der config.json des Modells",
    "Index": "aus den Kopfzeilen der Safetensors-/GGUF-Dateien (exakte Tensorgrößen)",
    "geschätzt": "aus der Geometrie gerechnet, nicht gemessen",
    "stat": "Dateigröße auf der Platte",
  };
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function json(url, opts) {
    // relative URL: die Seite wird auch unter einem Pfadpräfix (nginx /rigdash/) ausgeliefert
    const r = await fetch(url, Object.assign({ cache: "no-store" }, opts || {}));
    const text = await r.text();
    let j;
    try { j = JSON.parse(text); } catch (e) {
      throw new Error("HTTP " + r.status + ", keine JSON-Antwort von " + url + ": " + text.slice(0, 80));
    }
    if (!r.ok || j.ok === false) throw new Error(j.error || ("HTTP " + r.status));
    return j;
  }

  function liste() { return json("api/modellprofil/modelle"); }

  /* opts: {draft_path, kv_dtype: "auto"|"fp8_e4m3", mamba_ssm_dtype: "float32"|"bfloat16", gguf_file, registry: true|"kennung"} */
  function schaetzen(path, opts) {
    const body = Object.assign({ path: path }, opts || {});
    return json("api/modellprofil/schaetzen", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
  }

  function bytes(n) {
    if (n == null) return "–";
    const a = Math.abs(n);
    if (a >= 1073741824) return (n / 1073741824).toLocaleString("de-DE", { maximumFractionDigits: 2 }) + " GiB";
    if (a >= 1048576) return (n / 1048576).toLocaleString("de-DE", { maximumFractionDigits: 1 }) + " MiB";
    if (a >= 1024) return (n / 1024).toLocaleString("de-DE", { maximumFractionDigits: 1 }) + " KiB";
    return n.toLocaleString("de-DE") + " B";
  }
  const zahl = (n) => (n == null ? "–" : n.toLocaleString("de-DE"));
  const leaf = (o) => (o && typeof o === "object" && "v" in o ? o : { v: null, src: "" });

  /* Das Schätzprofil als flache Zeilenliste, gruppiert wie der Editor sie braucht. */
  function zeilen(p) {
    const out = [];
    const add = (gruppe, label, o, fmt, hinweis) => {
      const l = leaf(o);
      if (l.v == null || (fmt === bytes && l.v === 0)) return;          // Posten, den es im Modell nicht gibt
      const wert = fmt ? fmt(l.v) : String(l.v);
      if (wert == null) return;
      out.push({ gruppe: gruppe, label: label, wert: wert, roh: l.v, src: l.src, hinweis: hinweis || l.note || "" });
    };
    const a = p.arch || {}, w = p.weights || {}, kv = p.kv || {}, st = p.state || {}, ex = p.experts || {}, dr = p.draft || {}, cx = p.context || {};
    add("Modell", "Format", p.format, null, "Registry-Name; " + (leaf(p.format).src === "Index" ? "aus den Tensoren" : "aus der Config"));
    add("Modell", "Art", a.family, (v) => (v === "moe" ? "MoE" : "dicht"));
    add("Modell", "Layer", a.n_layers, zahl);
    const lc = leaf(a.layer_counts).v;
    if (lc) out.push({ gruppe: "Modell", label: "Layertypen", wert: "Attention " + lc.attn + " · GDN " + lc.gdn + (lc.mamba ? " · Mamba " + lc.mamba : ""), roh: lc, src: leaf(a.layer_counts).src, hinweis: "" });
    add("Modell", "Attention", a.attention, (v) => (v === "qsa" ? "QSA (Indexer)" : "voll"));
    add("Modell", "Hidden-Größe", a.hidden, zahl);
    add("Gewichte", "Gewichte gesamt", w.total_bytes, bytes, leaf(w.total_bytes).note);
    const fam = w.per_family_mean_bytes || {};
    add("Gewichte", "je Attention-Layer", fam.attn, bytes);
    add("Gewichte", "je GDN-Layer", fam.gdn, bytes);
    add("Gewichte", "je Mamba-Layer", fam.mamba, bytes);
    add("Gewichte", "Layer ohne Experten", w.layers_bytes_nonexpert, bytes);
    add("Gewichte", "Experten gesamt", w.layers_bytes_expert, (v) => (v ? bytes(v) : null));
    add("Gewichte", "Einbettung", w.embed_bytes, bytes);
    add("Gewichte", "lm_head", w.lm_head_bytes, bytes);
    add("Gewichte", "Sichtturm", w.visual_bytes, bytes, "wird bei language_model_only nicht geladen");
    add("Gewichte", "MTP-Kopf", w.mtp_bytes, bytes);
    const ple = (w.ple || {}).ngram_table_bytes;
    add("Gewichte", "n-gram-Tabelle", ple, bytes, "bleibt auf der Platte (mmap), erreicht das Gerät nie");
    add("KV", "KV je Token und Attention-Layer", kv.cell_bytes_per_attn_layer_token, (v) => zahl(v) + " B", "Wahl: " + leaf(kv.chosen).v);
    const vfp8 = ((kv.variants || {}).fp8_e4m3 || {}).bytes_per_token_all_attn_layers;
    add("KV", "KV je Token (fp8, alle Attention-Layer)", vfp8, (v) => zahl(v) + " B");
    add("Zustand", "Mamba/GDN-Zustand je Linear-Layer und Request", st.per_linear_layer_per_slot_mib, (v) => v.toLocaleString("de-DE", { maximumFractionDigits: 4 }) + " MiB",
        "SSM-Dtype " + leaf(st.ssm_dtype).v + (st.variants_mib ? " (float32 " + st.variants_mib.float32 + " / bfloat16 " + st.variants_mib.bfloat16 + " MiB)" : ""));
    if (leaf(ex.n).v) {
      add("Experten", "Anzahl", ex.n, zahl);
      add("Experten", "top_k", ex.top_k, zahl);
      add("Experten", "Bytes je Experte", ex.bytes_per_expert, bytes);
    }
    add("Draft", "MTP-Schichten", dr.mtp_layers, zahl);
    if (dr.external) add("Draft", "externer Draft", dr.external.total_bytes, bytes, (dr.external.architectures || {}).v ? dr.external.architectures.v.join(", ") : "");
    add("Kontext", "max. Positionen", cx.max_position_embeddings, zahl);
    const rp = (cx.rope || {});
    add("Kontext", "Rope", rp.type, (v) => v + (leaf(rp.theta).v ? " · θ " + zahl(leaf(rp.theta).v) : ""));
    add("Kontext", "Rope-erweitert", cx.rope_extended_tokens, zahl);
    add("Aktivierung", "Extend-Rate je Zeile", (p.activation || {}).extend_rate_mib_per_row, (v) => v.toLocaleString("de-DE", { maximumFractionDigits: 4 }) + " MiB",
        "Startwert aus der Geometrie; die Messung am Rang ersetzt ihn");
    return out;
  }

  function tabelle(p) {
    const rows = zeilen(p);
    let g = null, h = "<table class=\"mp-tab\"><thead><tr><th>Wert</th><th>Größe</th><th>Quelle</th></tr></thead><tbody>";
    rows.forEach((r) => {
      if (r.gruppe !== g) { g = r.gruppe; h += "<tr class=\"mp-grp\"><th colspan=\"3\">" + esc(g) + "</th></tr>"; }
      h += "<tr><td title=\"" + esc(r.hinweis) + "\">" + esc(r.label) + "</td><td>" + esc(r.wert) + "</td><td title=\"" + esc(QUELLE[r.src] || "") + "\">" + esc(r.src) + "</td></tr>";
    });
    return h + "</tbody></table>";
  }

  root.ModellProfil = { liste: liste, schaetzen: schaetzen, zeilen: zeilen, tabelle: tabelle, bytes: bytes, QUELLE: QUELLE };
  if (typeof module !== "undefined" && module.exports) module.exports = root.ModellProfil;
})(typeof window !== "undefined" ? window : (typeof globalThis !== "undefined" ? globalThis : this));
