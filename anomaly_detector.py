"""
Anomaly detection engine.

For each KPI we compute:
- deviation_from_target_pct : (value - target) / target * 100
- z_score                   : (value - rolling_mean) / rolling_std
                              using a 3-month trailing window of the
                              PRIOR months (excludes current to avoid
                              self-bias).
- severity                  : mapped from |z| and criticality weight.

A point is flagged when |z_score| >= 1.6 OR |deviation_from_target_pct| >= 8.
The threshold 1.6 is chosen so the demo's planted anomalies are caught
without flooding the alert panel with noise on the steady series.

Severity bucketing (uses direction-aware score so a 30% margin drop and
a 30% new-customer surge don't get treated identically):
- score = max(|z_score|, |dev_pct|/10) * criticality_weight
- score >= 7  -> High
- score >= 4  -> Medium
- otherwise   -> Low
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import statistics

from data.kpi_data import KPI, MONTHS, load_all_kpis


Z_THRESHOLD = 1.6          # flag if |z| at or above this
DEV_PCT_THRESHOLD = 8.0    # flag if abs deviation vs target at or above this
ROLLING_WINDOW = 3         # months of history used for rolling stats


@dataclass
class Anomaly:
    alert_id: str
    kpi_id: str
    kpi_name: str
    domain: str
    month: str
    month_index: int
    value: float
    target: float
    unit: str
    direction: str
    deviation_pct: float        # signed: + = above target, - = below
    z_score: float              # signed
    severity: str               # 'High' | 'Medium' | 'Low'
    severity_score: float
    criticality: int
    trigger_reasons: List[str] = field(default_factory=list)
    recent_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _rolling_stats(values: List[float], idx: int, window: int):
    """Return (mean, stdev) of the previous `window` values ending at idx-1.

    If there isn't enough history, falls back to whatever is available
    (or (0, 0) if nothing).
    """
    start = max(0, idx - window)
    prior = values[start:idx]
    if len(prior) < 2:
        # Not enough data for a meaningful stddev; return neutral stats
        # so z_score becomes 0 and we don't false-flag.
        return 0.0, 0.0
    mean = statistics.mean(prior)
    stdev = statistics.pstdev(prior)
    return mean, stdev


def _deviation_pct(value: float, target: float) -> float:
    if target == 0:
        return 0.0
    return (value - target) / target * 100.0


def _score_to_severity(score: float) -> str:
    if score >= 7:
        return "High"
    if score >= 4:
        return "Medium"
    return "Low"


def detect_anomaly_for_kpi(kpi: KPI, idx: int) -> Anomaly | None:
    """Inspect a single (kpi, month) pair. Returns Anomaly or None."""
    value = kpi.values[idx]
    target = kpi.targets[idx]
    mean, stdev = _rolling_stats(kpi.values, idx, ROLLING_WINDOW)
    dev_pct = _deviation_pct(value, target)
    z = ((value - mean) / stdev) if stdev > 0 else 0.0

    # Build the list of human-readable trigger reasons
    reasons: List[str] = []
    if abs(z) >= Z_THRESHOLD:
        side = "above" if z > 0 else "below"
        reasons.append(
            f"Value is {abs(z):.1f}\u03c3 {side} the {ROLLING_WINDOW}-month "
            f"rolling average ({mean:.2f})."
        )
    if abs(dev_pct) >= DEV_PCT_THRESHOLD:
        side = "above" if dev_pct > 0 else "below"
        reasons.append(
            f"Off target by {abs(dev_pct):.1f}% ({side} target of {target})."
        )

    flagged = (abs(z) >= Z_THRESHOLD) or (abs(dev_pct) >= DEV_PCT_THRESHOLD)
    if not flagged:
        return None

    # Direction-aware score: for 'lower_better' KPIs, an upward deviation
    # is worse; for 'higher_better', a downward deviation is worse. We
    # use the signed z and dev_pct to reflect this.
    if kpi.direction == "lower_better":
        bad_z = z
        bad_dev = dev_pct
    else:
        bad_z = -z
        bad_dev = -dev_pct

    # Use the worse of the two signals (max of positive contributions).
    z_mag = max(0.0, bad_z)
    dev_mag = max(0.0, bad_dev) / 10.0  # normalize to roughly z-scale
    base = max(z_mag, dev_mag)
    score = base * kpi.criticality

    # Recent history (last ROLLING_WINDOW months up to and including idx)
    hist_start = max(0, idx - ROLLING_WINDOW)
    history = []
    for h in range(hist_start, idx + 1):
        history.append({
            "month": MONTHS[h],
            "value": kpi.values[h],
            "target": kpi.targets[h],
        })

    return Anomaly(
        alert_id=f"{kpi.id}::{MONTHS[idx]}",
        kpi_id=kpi.id,
        kpi_name=kpi.name,
        domain=kpi.domain,
        month=MONTHS[idx],
        month_index=idx,
        value=value,
        target=target,
        unit=kpi.unit,
        direction=kpi.direction,
        deviation_pct=round(dev_pct, 2),
        z_score=round(z, 2),
        severity=_score_to_severity(score),
        severity_score=round(score, 2),
        criticality=kpi.criticality,
        trigger_reasons=reasons,
        recent_history=history,
    )


def detect_all_anomalies() -> List[Anomaly]:
    """Run detection across every KPI for every month.

    The intent for the demo is to surface issues in the most recent
    months. We still scan all months but the dashboard will focus on
    recent anomalies.
    """
    found: List[Anomaly] = []
    for kpi in load_all_kpis():
        for idx in range(len(kpi.values)):
            a = detect_anomaly_for_kpi(kpi, idx)
            if a is not None:
                found.append(a)
    # Sort newest first, then by severity_score desc.
    severity_rank = {"High": 3, "Medium": 2, "Low": 1}
    found.sort(key=lambda a: (a.month_index, severity_rank[a.severity],
                              a.severity_score), reverse=True)
    return found


def recent_anomalies(months_back: int = 2) -> List[Anomaly]:
    """Return only anomalies from the last N months (for the alert panel).

    Default is 2 months so the panel stays focused on the most recent
    issues - the demo dataset plants anomalies in Jul + Aug.
    """
    all_a = detect_all_anomalies()
    cutoff_idx = len(MONTHS) - months_back
    return [a for a in all_a if a.month_index >= cutoff_idx]


if __name__ == "__main__":
    for a in recent_anomalies():
        print(f"[{a.severity:6s}] {a.domain:10s} {a.kpi_name:28s} "
              f"{a.month}  val={a.value} tgt={a.target}  "
              f"z={a.z_score} dev={a.deviation_pct}%")