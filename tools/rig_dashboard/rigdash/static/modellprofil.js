/* Modellprofil schätzen (PROFIL-EDITOR S3, Auftrag 960): das kleine Modul hinter dem Knopf "Modellprofil erstellen".
   Rechnet NICHTS selbst: POST /api/modellprofil/schaetzen liefert das Modellprofil flliper.model/1 (der Schätzer liest nur
   config.json und die Kopfzeilen der Shards, nie ein Gewicht).  Dieses Modul holt es ab und macht daraus Zeilen
   {gruppe, label, wert, src, hinweis} -- die Oberfläche (Auftrag 930) zeichnet sie; tabelle() ist ein fertiger HTML-Baustein.
   Jeder Wert trägt seine Quelle: config | Index | geschätzt | stat.  Nur im Rig-Dashboard (Edition rig). */
(function (root) {
  "use strict";
  const QUELLE = {
    "config": "stated in the model's config.json",
    "Index": "from the header lines of the Safetensors/GGUF files (exact tensor sizes)",
    "geschätzt": "computed from the geometry, not measured",
    "stat": "file size on disk",
  };
  // display labels of the source tags (the tag itself is the API value and stays as is)
  const QUELLE_LABEL = { "geschätzt": "estimated" };
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function json(url, opts) {
    // relative URL: die Seite wird auch unter einem Pfadpräfix (nginx /rigdash/) ausgeliefert
    const r = await fetch(url, Object.assign({ cache: "no-store" }, opts || {}));
    const text = await r.text();
    let j;
    try { j = JSON.parse(text); } catch (e) {
      throw new Error("HTTP " + r.status + ", no JSON response from " + url + ": " + text.slice(0, 80));
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
    if (a >= 1073741824) return (n / 1073741824).toLocaleString("en-US", { maximumFractionDigits: 2 }) + " GiB";
    if (a >= 1048576) return (n / 1048576).toLocaleString("en-US", { maximumFractionDigits: 1 }) + " MiB";
    if (a >= 1024) return (n / 1024).toLocaleString("en-US", { maximumFractionDigits: 1 }) + " KiB";
    return n.toLocaleString("en-US") + " B";
  }
  const zahl = (n) => (n == null ? "–" : n.toLocaleString("en-US"));
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
    add("Model", "Format", p.format, null, "registry name; " + (leaf(p.format).src === "Index" ? "from the tensors" : "from the config"));
    add("Model", "Type", a.family, (v) => (v === "moe" ? "MoE" : "dense"));
    add("Model", "Layers", a.n_layers, zahl);
    const lc = leaf(a.layer_counts).v;
    if (lc) out.push({ gruppe: "Model", label: "Layer types", wert: "Attention " + lc.attn + " · GDN " + lc.gdn + (lc.mamba ? " · Mamba " + lc.mamba : ""), roh: lc, src: leaf(a.layer_counts).src, hinweis: "" });
    add("Model", "Attention", a.attention, (v) => (v === "qsa" ? "QSA (indexer)" : "full"));
    add("Model", "Hidden size", a.hidden, zahl);
    add("Weights", "Weights total", w.total_bytes, bytes, leaf(w.total_bytes).note);
    const fam = w.per_family_mean_bytes || {};
    add("Weights", "per attention layer", fam.attn, bytes);
    add("Weights", "per GDN layer", fam.gdn, bytes);
    add("Weights", "per Mamba layer", fam.mamba, bytes);
    add("Weights", "Layers without experts", w.layers_bytes_nonexpert, bytes);
    add("Weights", "Experts total", w.layers_bytes_expert, (v) => (v ? bytes(v) : null));
    add("Weights", "Embedding", w.embed_bytes, bytes);
    add("Weights", "lm_head", w.lm_head_bytes, bytes);
    add("Weights", "Vision tower", w.visual_bytes, bytes, "not loaded with language_model_only");
    add("Weights", "MTP head", w.mtp_bytes, bytes);
    const ple = (w.ple || {}).ngram_table_bytes;
    add("Weights", "n-gram table", ple, bytes, "stays on disk (mmap), never reaches the device");
    add("KV", "KV per token and attention layer", kv.cell_bytes_per_attn_layer_token, (v) => zahl(v) + " B", "choice: " + leaf(kv.chosen).v);
    const vfp8 = ((kv.variants || {}).fp8_e4m3 || {}).bytes_per_token_all_attn_layers;
    add("KV", "KV per token (fp8, all attention layers)", vfp8, (v) => zahl(v) + " B");
    add("State", "Mamba/GDN state per linear layer and request", st.per_linear_layer_per_slot_mib, (v) => v.toLocaleString("en-US", { maximumFractionDigits: 4 }) + " MiB",
        "SSM dtype " + leaf(st.ssm_dtype).v + (st.variants_mib ? " (float32 " + st.variants_mib.float32 + " / bfloat16 " + st.variants_mib.bfloat16 + " MiB)" : ""));
    if (leaf(ex.n).v) {
      add("Experts", "Count", ex.n, zahl);
      add("Experts", "top_k", ex.top_k, zahl);
      add("Experts", "Bytes per expert", ex.bytes_per_expert, bytes);
    }
    add("Draft", "MTP layers", dr.mtp_layers, zahl);
    if (dr.external) add("Draft", "external draft", dr.external.total_bytes, bytes, (dr.external.architectures || {}).v ? dr.external.architectures.v.join(", ") : "");
    add("Context", "max positions", cx.max_position_embeddings, zahl);
    const rp = (cx.rope || {});
    add("Context", "Rope", rp.type, (v) => v + (leaf(rp.theta).v ? " · θ " + zahl(leaf(rp.theta).v) : ""));
    add("Context", "Rope extended", cx.rope_extended_tokens, zahl);
    add("Activation", "Extend rate per row", (p.activation || {}).extend_rate_mib_per_row, (v) => v.toLocaleString("en-US", { maximumFractionDigits: 4 }) + " MiB",
        "initial value from the geometry; the measurement on the rank replaces it");
    return out;
  }

  function tabelle(p) {
    const rows = zeilen(p);
    let g = null, h = "<table class=\"mp-tab\"><thead><tr><th>Value</th><th>Size</th><th>Source</th></tr></thead><tbody>";
    rows.forEach((r) => {
      if (r.gruppe !== g) { g = r.gruppe; h += "<tr class=\"mp-grp\"><th colspan=\"3\">" + esc(g) + "</th></tr>"; }
      h += "<tr><td title=\"" + esc(r.hinweis) + "\">" + esc(r.label) + "</td><td>" + esc(r.wert) + "</td><td title=\"" + esc(QUELLE[r.src] || "") + "\">" + esc(QUELLE_LABEL[r.src] || r.src) + "</td></tr>";
    });
    return h + "</tbody></table>";
  }

  root.ModellProfil = { liste: liste, schaetzen: schaetzen, zeilen: zeilen, tabelle: tabelle, bytes: bytes, QUELLE: QUELLE };
  if (typeof module !== "undefined" && module.exports) module.exports = root.ModellProfil;
})(typeof window !== "undefined" ? window : (typeof globalThis !== "undefined" ? globalThis : this));
