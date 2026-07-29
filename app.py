"""
Flask app for the Intelligent Early Warning System.

Endpoints
---------
GET  /                       - Dashboard HTML page
GET  /api/bootstrap          - All data the dashboard needs on first load:
                               KPIs per domain, anomaly list, and (if
                               already analyzed) the synthesis summary.
GET  /api/analysis           - Per-alert root-cause + recommended action
                               plus the executive synthesis. Computed on
                               first call, cached in-memory.
POST /api/refresh-analysis   - Re-run the LLM agents. Useful for demos
                               after changing the API key.

The app is intentionally small and dependency-light (just flask + requests).
Run with:  python app.py
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
from typing import Dict, Any

from flask import Flask, jsonify, render_template, request

from data.kpi_data import kpis_by_domain, load_all_kpis, MONTHS
from anomaly_detector import detect_all_anomalies, recent_anomalies
from llm_client import get_client
import agents


app = Flask(__name__)


# ---------- in-memory caches -------------------------------------------

_ANALYSIS_CACHE: Dict[str, Any] | None = None


def _run_analysis() -> Dict[str, Any]:
    """Run the full agent pipeline once and cache the result."""
    global _ANALYSIS_CACHE
    if _ANALYSIS_CACHE is not None:
        return _ANALYSIS_CACHE

    # IMPORTANT: analyze the SAME anomalies that the alerts panel shows
    # (months_back=1) so we never make more LLM calls than the dashboard
    # has visible alerts. With months_back=2 we'd run ~15 per-alert calls
    # + 1 synthesis = 16 sequential real-API calls; on Groq's free tier
    # that's enough to trip 429 in the middle of a judged demo. Sticking
    # to months_back=1 keeps us at 8-9 calls (~22s wall-clock at 2.5s
    # throttle), well inside the 30 req/min budget.
    anomalies = recent_anomalies(months_back=1)
    if not anomalies:
        # Defensive: if nothing flagged, still show a friendly message
        anomalies = detect_all_anomalies()

    result = agents.analyze_all(anomalies)
    _ANALYSIS_CACHE = result
    return result


def _bootstrap_payload() -> Dict[str, Any]:
    """Compose the data the frontend loads on first render."""
    by_domain = kpis_by_domain()
    domains_payload = {}
    for domain, kpis in by_domain.items():
        domains_payload[domain] = [
            {
                "id": k.id,
                "name": k.name,
                "unit": k.unit,
                "direction": k.direction,
                "criticality": k.criticality,
                "months": MONTHS,
                "values": k.values,
                "targets": k.targets,
            }
            for k in kpis
        ]

    anomalies = recent_anomalies(months_back=1)
    if not anomalies:
        anomalies = detect_all_anomalies()

    alerts_payload = []
    for a in anomalies:
        alerts_payload.append({
            "alert_id": a.alert_id,
            "kpi_id": a.kpi_id,
            "kpi_name": a.kpi_name,
            "domain": a.domain,
            "month": a.month,
            "value": a.value,
            "target": a.target,
            "unit": a.unit,
            "deviation_pct": a.deviation_pct,
            "z_score": a.z_score,
            "severity": a.severity,
            "severity_score": a.severity_score,
            "criticality": a.criticality,
            "trigger_reasons": a.trigger_reasons,
            "recent_history": a.recent_history,
            "direction": a.direction,
        })

    # Eagerly trigger analysis on first load so the synthesis and
    # per-alert analysis are ready when the user clicks.
    analysis = _run_analysis()

    return {
        "months": MONTHS,
        "domains": domains_payload,
        "alerts": alerts_payload,
        "analysis": analysis,
    }


# ---------- routes ------------------------------------------------------

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/bootstrap")
def api_bootstrap():
    return jsonify(_bootstrap_payload())


@app.route("/api/analysis")
def api_analysis():
    return jsonify(_run_analysis())


@app.route("/api/refresh-analysis", methods=["POST"])
def api_refresh_analysis():
    global _ANALYSIS_CACHE
    _ANALYSIS_CACHE = None
    return jsonify({"ok": True, "analysis": _run_analysis()})


@app.route("/api/health")
def api_health():
    client = get_client()
    return jsonify({
        "status": "ok",
        "model": client.model,
        "base_url": client.base_url,
        "real_api": client.use_real,
        "kpi_count": len(load_all_kpis()),
    })


# ---------- entrypoint --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # debug=False so we don't double-run the analysis; the user can flip
    # it during development if they want auto-reload.
    app.run(host="0.0.0.0", port=port, debug=False)