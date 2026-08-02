/* =====================================================================
   Argus - frontend logic
   Loads /api/bootstrap on startup, renders 3 charts + alert list,
   lazily renders per-alert agent analysis on click.
   ===================================================================== */

(function () {
  "use strict";

  const SEVERITY_ORDER = { High: 0, Medium: 1, Low: 2 };
  const SEVERITY_RANK_CLASS = {
    High:   "badge-danger",
    Medium: "badge-warning",
    Low:    "badge-info",
  };
  const DOMAIN_COLORS = {
    Finance:    { line: "#2563eb", fill: "rgba(37,99,235,.08)"  },
    Sales:      { line: "#10b981", fill: "rgba(16,185,129,.08)"  },
    Operations: { line: "#6366f1", fill: "rgba(99,102,241,.08)" },
  };

  // ---------- boot -----------------------------------------------------

  document.addEventListener("DOMContentLoaded", async () => {
    document.getElementById("refresh-btn")
            .addEventListener("click", onRefreshClick);

    // Wire up the per-chart "Reset zoom" buttons (delegated handler).
    document.querySelectorAll(".reset-zoom-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-zoom-target");
        const inst = _chartInstances[id];
        if (inst && inst.resetZoom) inst.resetZoom();
      });
    });

    // Part 2: open the chart-expand modal when the user clicks anywhere
    // on a chart card (except the inline reset-zoom button). The handler
    // is delegated at the grid level so it doesn't touch the inline
    // chart code at all.
    document.querySelector(".charts-grid")
      ?.addEventListener("click", onChartCardClick);
    document.querySelector(".charts-grid")
      ?.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          const card = ev.target.closest(".chart-card-clickable");
          if (card) {
            ev.preventDefault();
            openChartModal(card.getAttribute("data-chart-id"),
                           card.getAttribute("data-domain"));
          }
        }
      });

    // Part 2: modal close handlers (X button, backdrop click, Escape).
    document.getElementById("chart-modal-close")
      ?.addEventListener("click", closeChartModal);
    document.getElementById("chart-modal-reset")
      ?.addEventListener("click", () => {
        const inst = _modalChartInstance;
        if (inst && inst.resetZoom) inst.resetZoom();
      });
    document.querySelectorAll("[data-modal-close]").forEach(el => {
      el.addEventListener("click", closeChartModal);
    });
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") closeChartModal();
    });

    try {
      const payload = await fetchJSON("/api/bootstrap");
      renderModelBadge(payload.analysis && payload.analysis.model_info);
      renderErrorBanner(payload.analysis);
      renderSynthesis(payload.analysis, payload.alerts);
      renderAlerts(payload.alerts, payload.analysis);
      renderCharts(payload.domains, payload.months, payload.alerts);
      document.getElementById("alert-count-badge").textContent =
        payload.alerts.length + " alerts";
    } catch (err) {
      console.error(err);
      document.getElementById("alerts-list").innerHTML =
        `<div class="placeholder">Failed to load dashboard: ${escapeHtml(String(err))}</div>`;
    }
  });

  // ---------- helpers --------------------------------------------------

  async function fetchJSON(url, options) {
    const resp = await fetch(url, options);
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    return resp.json();
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  function fmtValue(v, unit) {
    if (typeof v !== "number") return v;
    if (unit === "%" || unit === "$M" || unit === "$k") return v.toFixed(2);
    if (unit === "k units" || unit === "count") return Math.round(v).toLocaleString();
    return v.toFixed(2);
  }

  function fmtPct(v) {
    const sign = v > 0 ? "+" : "";
    return `${sign}${v.toFixed(1)}%`;
  }

  // ---------- model badge ----------------------------------------------

  function renderModelBadge(modelInfo) {
    const el = document.getElementById("model-badge");
    if (!modelInfo) { el.textContent = "model: unknown"; return; }
    if (modelInfo.real_api) {
      el.classList.add("live");
      el.textContent = `LLM: ${modelInfo.model}`;
    } else {
      el.classList.add("demo");
      el.textContent = `Demo mode · ${modelInfo.model}`;
    }
  }

  // ---------- error banner (visible when live LLM is throttled) -------

  function renderErrorBanner(analysis) {
    // Remove any existing banner so refreshes don't stack them.
    const old = document.getElementById("llm-error-banner");
    if (old) old.remove();

    if (!analysis || !analysis.has_errors) return;

    const stats = analysis.session_stats || {};
    const rateLimited = (stats.calls_rate_limited || 0) > 0;
    const failed = (stats.calls_failed || 0) > 0;

    let msg;
    if (rateLimited) {
      msg = `Live LLM hit Groq's rate limit (${stats.calls_rate_limited} ` +
            `calls throttled, ${stats.calls_retried || 0} retried). ` +
            `The free tier allows ~30 req/min; affected agents fell back ` +
            `to a brief notice. Wait a minute, then click ⟳ Refresh analysis.`;
    } else if (failed) {
      msg = `Some live LLM calls failed (${stats.calls_failed}). ` +
            `Affected agents show a notice in place of an analysis. ` +
            `Click ⟳ Refresh analysis to retry.`;
    } else {
      return;
    }

    const banner = document.createElement("div");
    banner.id = "llm-error-banner";
    banner.className = "llm-banner";
    banner.innerHTML = `
      <span class="llm-banner-icon">⚠</span>
      <span class="llm-banner-text">${escapeHtml(msg)}</span>
      <button class="llm-banner-close" aria-label="Dismiss">×</button>
    `;
    banner.querySelector(".llm-banner-close")
          .addEventListener("click", () => banner.remove());

    // Insert just above the synthesis card.
    const synth = document.getElementById("synthesis-section");
    synth.parentNode.insertBefore(banner, synth);
  }

  // ---------- synthesis -----------------------------------------------

  function renderSynthesis(analysis, alerts) {
    const body = document.getElementById("synthesis-body");
    if (!analysis || !analysis.synthesis) {
      body.innerHTML = `<div class="placeholder">No synthesis available.</div>`;
      return;
    }
    const s = analysis.synthesis;
    const actions = (s.prioritized_actions || []).slice(0, 3);

    body.innerHTML = `
      <div class="synthesis-summary">${escapeHtml(s.executive_summary || "(no summary)")}</div>
      <div class="synthesis-actions">
        ${actions.map((a, i) => `
          <div class="synthesis-action">
            <div class="action-rank">Priority ${i + 1}</div>
            ${escapeHtml(a)}
          </div>
        `).join("")}
      </div>
    `;
  }

  // ---------- alert list ----------------------------------------------

  function renderAlerts(alerts, analysis) {
    const list = document.getElementById("alerts-list");
    if (!alerts.length) {
      list.innerHTML = `<div class="placeholder">No anomalies flagged in the last 3 months. All KPIs are within tolerance.</div>`;
      return;
    }

    // Sort: severity desc (High first), then severity_score desc, then newest.
    const sorted = alerts.slice().sort((a, b) => {
      const s = SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity];
      if (s !== 0) return s;
      if (b.severity_score !== a.severity_score) return b.severity_score - a.severity_score;
      return a.month < b.month ? 1 : -1;
    });

    list.innerHTML = sorted.map(a => alertSummaryHTML(a)).join("");

    sorted.forEach(a => {
      const node = document.querySelector(`[data-alert-id="${a.alert_id}"]`);
      if (!node) return;
      node.querySelector(".alert-summary")
          .addEventListener("click", () => toggleAlert(node, a, analysis));
    });
  }

  function alertSummaryHTML(a) {
    const devClass = isBadDeviation(a) ? "bad" : "";
    return `
      <article class="alert severity-${a.severity}" data-alert-id="${escapeHtml(a.alert_id)}">
        <div class="alert-summary">
          <span class="badge ${SEVERITY_RANK_CLASS[a.severity]}">${a.severity}</span>
          <div>
            <div class="alert-domain">${escapeHtml(a.domain)} · ${escapeHtml(a.month)}</div>
            <div class="alert-title">
              ${escapeHtml(a.kpi_name)}
              <small>criticality ${a.criticality}/3</small>
            </div>
          </div>
          <div class="alert-dev ${devClass}">
            <strong>${fmtValue(a.value, a.unit)}</strong>
            ${a.unit} · target ${fmtValue(a.target, a.unit)}
            · <strong>${fmtPct(a.deviation_pct)}</strong>
          </div>
          <span class="alert-chevron">›</span>
        </div>
        <div class="alert-body" hidden>
          <div class="placeholder">Loading analysis…</div>
        </div>
      </article>
    `;
  }

  function isBadDeviation(a) {
    // For higher_better: negative deviation is bad.
    // For lower_better: positive deviation is bad.
    if (a.direction === "lower_better") return a.deviation_pct > 0;
    return a.deviation_pct < 0;
  }

  function toggleAlert(node, alert, analysis) {
    const isOpen = node.classList.toggle("open");
    const body = node.querySelector(".alert-body");
    if (!isOpen) { body.hidden = true; return; }
    body.hidden = false;
    if (body.dataset.loaded === "1") return;
    renderAlertBody(body, alert, analysis);
    body.dataset.loaded = "1";
  }

  function renderAlertBody(body, alert, analysis) {
    const cached = analysis && analysis.per_alert && analysis.per_alert[alert.alert_id];
    const triggers = alert.trigger_reasons.map(t => `<li>${escapeHtml(t)}</li>`).join("");

    // Build history table
    const historyRows = alert.recent_history.map(h => {
      const isCurrent = h.month === alert.month;
      const dev = alert.target !== 0 ? ((h.value - alert.target) / alert.target) * 100 : 0;
      const devClass = Math.abs(dev) >= 8 ? "deviation" : "";
      return `
        <tr class="${isCurrent ? "current" : ""} ${devClass}">
          <td>${escapeHtml(h.month)}${isCurrent ? " (current)" : ""}</td>
          <td>${fmtValue(h.value, alert.unit)} ${escapeHtml(alert.unit)}</td>
          <td>${fmtValue(h.target, alert.unit)} ${escapeHtml(alert.unit)}</td>
          <td>${fmtPct(dev)}</td>
        </tr>
      `;
    }).join("");

    body.innerHTML = `
      <h4>What triggered this</h4>
      <ul class="trigger-list">${triggers}</ul>

      <h4>Recent data points</h4>
      <table class="history-table">
        <thead><tr><th>Month</th><th>Value</th><th>Target</th><th>Δ vs target</th></tr></thead>
        <tbody>${historyRows}</tbody>
      </table>

      <h4>Root cause &amp; recommended action</h4>
      <div id="agent-block-${escapeHtml(alert.alert_id)}" class="agent-block loading">
        ${cached
          ? renderAgentBlock(cached)
          : "Analysis not cached - run a refresh to populate."}
      </div>
    `;
  }

  function renderAgentBlock(a) {
    const errTag = a._error
      ? `<div class="agent-error-tag">⚠ LLM unavailable - retry shortly</div>`
      : "";
    return `
      <div class="agent-label">${escapeHtml(a.domain)} Expert Agent</div>
      <div class="agent-text">${escapeHtml(a.root_cause)}</div>
      <div class="agent-action">
        <div class="action-label">Recommended next action</div>
        <div>${escapeHtml(a.recommended_action)}</div>
      </div>
      ${errTag}
    `;
  }

  // ---------- charts --------------------------------------------------

  let _chartInstances = {};

  function renderCharts(domains, months, alerts) {
    Object.entries(domains).forEach(([domain, kpis]) => {
      const canvasId = `chart-${domain.toLowerCase()}`;
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;

      const colors = DOMAIN_COLORS[domain] || DOMAIN_COLORS.Finance;

      // Anomaly month indices by kpi_id (for marker dots)
      const anomalyIdxByKpi = {};
      alerts.filter(a => a.domain === domain).forEach(a => {
        anomalyIdxByKpi[a.kpi_id] = a.month;
      });

      const datasets = [];
      // Build actuals first, targets second. Each dataset carries a
      // direct reference to its KPI so the tooltip can read the correct
      // unit/direction without any index arithmetic.
      kpis.forEach((kpi, i) => {
        const color = shiftColor(colors.line, i, kpis.length);
        datasets.push({
          label: `${kpi.name} (actual)`,
          data: kpi.values,
          borderColor: color,
          backgroundColor: colors.fill,
          borderWidth: 2,
          tension: 0.25,
          _kpi: kpi,
          pointRadius: (ctx) => {
            const idx = ctx.dataIndex;
            return anomalyIdxByKpi[kpi.id] === months[idx] ? 6 : 3;
          },
          pointBackgroundColor: (ctx) => {
            const idx = ctx.dataIndex;
            return anomalyIdxByKpi[kpi.id] === months[idx] ? "#ef4444" : color;
          },
          pointBorderColor: () => "#fff",
          pointBorderWidth: (ctx) => {
            const idx = ctx.dataIndex;
            return anomalyIdxByKpi[kpi.id] === months[idx] ? 2 : 1;
          },
          fill: false,
        });
      });

      // Dashed target lines (one per KPI), thin and faint
      kpis.forEach((kpi, i) => {
        const color = shiftColor(colors.line, i, kpis.length);
        datasets.push({
          label: `${kpi.name} (target)`,
          data: kpi.targets,
          borderColor: color,
          borderDash: [4, 4],
          borderWidth: 1.2,
          pointRadius: 0,
          fill: false,
          _kpi: kpi,
        });
      });

      // Tear down any previous instance for this canvas.
      const prev = _chartInstances[canvasId];
      if (prev) prev.destroy();

      _chartInstances[canvasId] = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: { labels: months, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: {
              position: "bottom",
              labels: { boxWidth: 10, font: { size: 11 }, color: "#374151" },
            },
            tooltip: {
              callbacks: {
                // Use the dataset's own _kpi ref so units and values
                // always match the line being hovered.
                label: (ctx) => {
                  const v = ctx.parsed.y;
                  const kpi = ctx.dataset._kpi || {};
                  return `${ctx.dataset.label}: ${fmtValue(v, kpi.unit)} ${kpi.unit || ""}`.trim();
                },
              },
            },
            // chartjs-plugin-zoom: scroll-wheel zoom on the X axis,
            // shift-drag to pan, double-click resets.
            zoom: {
              pan: { enabled: true, mode: "x", modifierKey: "shift" },
              zoom: {
                wheel: { enabled: true, modifierKey: null },
                pinch: { enabled: true },
                drag: { enabled: false },
                mode: "x",
              },
              limits: { x: { min: 0, max: months.length - 1 } },
            },
          },
          scales: {
            x: {
              grid: { color: "#f1f5f9" },
              ticks: { color: "#6b7280", font: { size: 11 } },
            },
            y: {
              grid: { color: "#f1f5f9" },
              ticks: { color: "#6b7280", font: { size: 11 } },
            },
          },
          onDoubleClick: (evt) => {
            // Native double-click resets zoom too.
            const items = _chartInstances[canvasId] && evt.chart;
            if (items && items.resetZoom) items.resetZoom();
          },
        },
      });

      // Update domain meta with anomaly count
      const meta = document.getElementById(
        `${domain.toLowerCase() === "operations" ? "ops" : domain.toLowerCase()}-meta`
      );
      if (meta) {
        const n = alerts.filter(a => a.domain === domain).length;
        meta.textContent = n > 0
          ? `${n} alert${n > 1 ? "s" : ""} flagged`
          : "within tolerance";
      }
    });
  }

  // Generate a slight color shift per KPI within a domain
  function shiftColor(hex, idx, total) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const delta = total > 1 ? Math.round(40 * (idx / (total - 1)) - 20) : 0;
    const clamp = v => Math.max(0, Math.min(255, v + delta));
    return `rgb(${clamp(r)}, ${clamp(g)}, ${clamp(b)})`;
  }

  // ---------- refresh -------------------------------------------------

  async function onRefreshClick(ev) {
    const btn = ev.currentTarget;
    btn.disabled = true;
    const origLabel = btn.textContent;
    btn.textContent = "⟳ Re-running agents… (may take ~30s on free tier)";
    try {
      const result = await fetchJSON("/api/refresh-analysis", { method: "POST" });
      renderModelBadge(result.analysis.model_info);
      renderErrorBanner(result.analysis);
      renderSynthesis(result.analysis, []);
      // Re-render visible alert bodies with fresh analysis.
      document.querySelectorAll(".alert.open").forEach(node => {
        const alertId = node.dataset.alert_id || node.getAttribute("data-alert-id");
        const block = document.getElementById(`agent-block-${alertId}`);
        if (block && result.analysis.per_alert[alertId]) {
          block.classList.remove("loading");
          block.innerHTML = renderAgentBlock(result.analysis.per_alert[alertId]);
        }
      });
    } catch (err) {
      console.error(err);
      alert("Refresh failed: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = origLabel;
    }
  }

  // ---------- chart expand modal (Part 2) -----------------------------
  // Keeps the modal chart instance in a module-scoped var so the reset
  // button and Escape handler can reach it. Independent of the inline
  // charts entirely: it builds its own Chart from the same dataset
  // shape so the inline charts aren't touched when the modal opens.

  let _modalChartInstance = null;

  function onChartCardClick(ev) {
    // Ignore clicks on the inline reset-zoom button (its own handler).
    if (ev.target.closest(".reset-zoom-btn")) return;
    const card = ev.target.closest(".chart-card-clickable");
    if (!card) return;
    openChartModal(card.getAttribute("data-chart-id"),
                   card.getAttribute("data-domain"));
  }

  function openChartModal(canvasId, domainLabel) {
    const inline = _chartInstances[canvasId];
    if (!inline) return;

    const modal = document.getElementById("chart-modal");
    const canvas = document.getElementById("chart-modal-canvas");
    const title = document.getElementById("chart-modal-title");
    if (!modal || !canvas) return;

    title.textContent = `${domainLabel} — KPI Trend`;

    // Tear down any previous modal chart.
    if (_modalChartInstance) {
      _modalChartInstance.destroy();
      _modalChartInstance = null;
    }

    // Build a deep-cloned copy of the inline datasets so the modal chart
    // never mutates the inline one.
    const clonedData = {
      labels: inline.data.labels.slice(),
      datasets: inline.data.datasets.map(d => ({
        ...d,
        data: d.data.slice(),
        pointRadius: typeof d.pointRadius === "function" ? 4 : d.pointRadius,
        pointBackgroundColor: typeof d.pointBackgroundColor === "function"
          ? "#ef4444" : d.pointBackgroundColor,
        pointBorderColor: "#fff",
        pointBorderWidth: 1,
      })),
    };

    _modalChartInstance = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: clonedData,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "bottom",
            labels: { boxWidth: 12, font: { size: 12 }, color: "#374151" },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const v = ctx.parsed.y;
                const kpi = ctx.dataset._kpi || {};
                return `${ctx.dataset.label}: ${fmtValue(v, kpi.unit)} ${kpi.unit || ""}`.trim();
              },
            },
          },
          // chartjs-plugin-zoom v2: wheel + pinch zoom, drag to pan,
          // double-click to reset. Larger drag area in the popup.
          zoom: {
            pan:  { enabled: true, mode: "xy" },
            zoom: {
              wheel: { enabled: true },
              pinch: { enabled: true },
              drag:  { enabled: false },
              mode:  "xy",
            },
          },
        },
        scales: {
          x: {
            grid: { color: "#f1f5f9" },
            ticks: { color: "#6b7280", font: { size: 12 } },
          },
          y: {
            grid: { color: "#f1f5f9" },
            ticks: { color: "#6b7280", font: { size: 12 } },
          },
        },
        onDoubleClick: (evt) => {
          const c = _modalChartInstance;
          if (c && c.resetZoom) c.resetZoom();
        },
      },
    });

    modal.hidden = false;
    modal.setAttribute("aria-hidden", "false");
  }

  function closeChartModal() {
    const modal = document.getElementById("chart-modal");
    if (!modal || modal.hidden) return;
    modal.hidden = true;
    modal.setAttribute("aria-hidden", "true");
    if (_modalChartInstance) {
      _modalChartInstance.destroy();
      _modalChartInstance = null;
    }
  }

})();