"""Erzeugt die Grafana-Tafel "Rig – Verlauf" (rig-verlauf.json) aus VictoriaMetrics-PromQL.

Nutzer 01.10.: Verläufe aus der Zeitreihen-DB, Grafana direkt daneben; der rigdash bleibt für den
Live-Zustand.  Eine Quelle der Wahrheit für die Tafel ist diese Datei; nach Änderung
``python3 make_dashboard.py > dashboards/rig-verlauf.json`` und ``install_grafana.sh``.
"""

import json

DS = {"type": "prometheus", "uid": "vm"}
_id = [0]


def panel(title, targets, unit="short", x=0, y=0, w=12, h=8, desc="", draw="line", stack=False, decimals=None,
          kind="timeseries", min0=True, interval=None):
    _id[0] += 1
    fc = {"unit": unit, "custom": {"drawStyle": draw, "lineWidth": 1 if draw == "line" else 0, "fillOpacity": 12,
                                   "pointSize": 6, "showPoints": "always" if draw == "points" else "never",
                                   "spanNulls": False, "stacking": {"mode": "normal" if stack else "none"}}}
    if min0:
        fc["min"] = 0
    if decimals is not None:
        fc["decimals"] = decimals
    p = {"id": _id[0], "type": kind, "title": title, "description": desc, "datasource": DS,
         "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "fieldConfig": {"defaults": fc, "overrides": []},
         "options": {"legend": {"displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull", "mean", "max"]},
                     "tooltip": {"mode": "multi", "sort": "desc"}},
         "targets": [dict({"datasource": DS, "refId": chr(65 + i)}, **t) for i, t in enumerate(targets)]}
    if interval:
        p["interval"] = interval
    if kind == "stat":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": "none", "graphMode": "area", "textMode": "value_and_name"}
    return p


def t(expr, legend):
    return {"expr": expr, "legendFormat": legend}


def row(title, y):
    _id[0] += 1
    return {"id": _id[0], "type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
            "panels": []}


# Nutzer 01.10. (Nachtrag): TTFT und Flipzeit sind Einzelereignisse -> Punkte, keine Linien. Bis die Front je
# Anfrage einen Punkt liefert (weg2_req, Feldwunsch), ist ein Punkt = die Anfragen eines 5-s-Takts der Bruecke.
TTFT_MEAN = ("sum by (model) (increase(weg2_front_ttft_ms_sum{model=~\"$model\"}[5s])) / "
             "(sum by (model) (increase(weg2_front_ttft_count{model=~\"$model\"}[5s])) > 0) / 1000")
GPU_NAME = "* on (uuid) group_left (name, index) nvidia_smi_gpu_info"

panels = [
    row("Nutzer-Latenz", 0),
    panel("TTFT der Nutzer (Punkt je 5-s-Takt)", [t(TTFT_MEAN, "{{model}}")], "s", 0, 1, 12, 8,
          "Ankunft an der Front -> erster Inhalt von D (front.py LEG2-FIRST-CONTENT), IPC-Zähler "
          "state.json front.arrival_seat.ttft_* (NF). Mittel der Anfragen, deren erstes Token im Intervall kam. "
          "27B: fehlt bis zum Image mit weg2_ttft_seconds (TSDB-Delta 01.10.). Über P / direkt D getrennt: "
          "fehlt in IPC (front.arrival_seat.ttft_by_via).", draw="points", interval="5s"),
    panel("TTFT max seit Boot / Anfragen je min", [
        t("max by (model) (weg2_front_ttft_ms_max{model=~\"$model\"}) / 1000", "max {{model}}"),
    ], "s", 12, 1, 6, 8, "state.json front.arrival_seat.ttft_ms_max"),
    panel("Anfragen bedient je min (D-Bein)", [
        t("sum by (model) (rate(weg2_front_served_total{group=\"D\",model=~\"$model\"}[$__rate_interval])) * 60", "{{model}}")],
        "short", 18, 1, 6, 8, "state.json front.served.D"),
    panel("Flipzeit (je Flip)", [
        t("max by (model, dir) (weg2_flip_user_view_ms{part=\"total\",model=~\"$model\"}) / 1000", "{{dir}} {{model}}"),
        t("max by (model, dir) (weg2_flip_user_view_ms{part=\"layer\",model=~\"$model\"}) / 1000", "Layer-Tausch {{dir}} {{model}}")],
        "s", 0, 9, 12, 8, "Nutzersicht (01.10.): P→D = P-Ende -> erstes Decode auf D (Rang-Segmente), "
        "D→P = Decode-Ende -> P-Prefill-Start (flip_user_time); Teile part=vorlauf|layer|nachlauf|d_extend. "
        "Ein Punkt je Flip zum flip_begin; Leerlauf-Flips ohne Wert.", draw="points"),
    panel("Upstream-TTFT des D-Beins (nicht Nutzer-TTFT)", [
        t("histogram_quantile(0.5, sum by (le) (rate(sglang:time_to_first_token_seconds_bucket[$__rate_interval])))", "p50"),
        t("histogram_quantile(0.9, sum by (le) (rate(sglang:time_to_first_token_seconds_bucket[$__rate_interval])))", "p90")],
        "s", 12, 9, 12, 8, "sglang:time_to_first_token_seconds der wachen Gruppe (Front-Durchreiche /metrics): "
        "ab Ankunft des Leg 2 bei der Gruppe, ohne Warteschlange und Flip der Front."),
    row("Durchsatz", 17),
    panel("Decode tok/s (D, TP0)", [
        t("sum by (model) (rate(weg2_rank_decode_tokens_total{group=\"D\",rank=\"tp0pp0\",model=~\"$model\"}[$__rate_interval]))", "{{model}}")],
        "short", 0, 18, 12, 8, "rankstats decode.tokens (Wanduhr-Rate, Pausen zählen mit)"),
    panel("Prefill neu gerechnet tok/s (P, D)", [
        t("sum by (model, group) (rate(weg2_rank_prefill_new_tokens_total{rank=\"tp0pp0\",group=~\"P|D\",model=~\"$model\"}[$__rate_interval]))", "{{group}} {{model}}")],
        "short", 12, 18, 12, 8, "rankstats prefill.new_tokens, erster Rang je Gruppe (Wanduhr-Rate)"),
    panel("Decode bs (letzte Runde) / laufende Anfragen", [
        t("max by (model) (weg2_rank_decode_last_bs{group=\"D\",rank=\"tp0pp0\",model=~\"$model\"})", "bs {{model}}"),
        t("max by (model) (weg2_front_outstanding_n{model=~\"$model\"})", "outstanding {{model}}"),
        t("max by (model) (weg2_front_queue{model=~\"$model\"})", "Warteschlange {{model}}")],
        "short", 0, 26, 12, 8),
    panel("KV-Belegung", [
        t("max by (model, group) (weg2_rank_kv_usage_ratio{model=~\"$model\"}) * 100", "{{group}} {{model}}")],
        "percent", 12, 26, 12, 8, "rankstats sched.full_token_usage"),
    row("Karten und Host", 34),
    panel("Leistungsaufnahme", [t("sum(nvidia_smi_power_draw_watts)", "Summe"),
                                t("nvidia_smi_power_draw_watts " + GPU_NAME, "nvml {{index}} {{name}}")],
          "watt", 0, 35, 12, 8, "nvidia_gpu_exporter (NVML)"),
    panel("GPU-Temperatur", [t("nvidia_smi_temperature_gpu " + GPU_NAME, "nvml {{index}} {{name}}")],
          "celsius", 12, 35, 12, 8, min0=False),
    panel("GPU-Last / VRAM belegt", [t("nvidia_smi_utilization_gpu_ratio * 100 " + GPU_NAME, "Last nvml {{index}}"),
                                     t("nvidia_smi_memory_used_bytes / nvidia_smi_memory_total_bytes * 100 " + GPU_NAME, "VRAM nvml {{index}}")],
          "percent", 0, 43, 12, 8),
    panel("Proxmox-Host: RAM verfügbar und ZFS-ARC", [
        t("node_memory_MemAvailable_bytes{host=\"proxmox\"}", "MemAvailable"),
        t("node_zfs_arc_size{host=\"proxmox\"}", "ZFS-ARC"),
        t("node_memory_SwapTotal_bytes{host=\"proxmox\"} - node_memory_SwapFree_bytes{host=\"proxmox\"}", "Swap belegt")],
        "bytes", 12, 43, 12, 8, "node_exporter auf dem Proxmox-Host (192.168.0.11:9100)"),
]

dash = {
    "uid": "rig-verlauf", "title": "Rig – Verlauf", "tags": ["rig", "weg2"], "timezone": "browser",
    "schemaVersion": 39, "version": 1, "refresh": "10s", "time": {"from": "now-6h", "to": "now"},
    "graphTooltip": 1, "editable": True,
    "templating": {"list": [{"name": "model", "label": "Modell", "type": "custom", "query": "NF,27B",
                             "current": {"text": "All", "value": "$__all"}, "includeAll": True, "allValue": ".*",
                             "multi": False, "options": []}]},
    "panels": panels,
}
print(json.dumps(dash, indent=1, ensure_ascii=False))
