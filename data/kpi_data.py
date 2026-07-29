"""
Synthetic KPI data for a mid-size company across 3 domains.

8 months of monthly values (Jan 2026 - Aug 2026) for 8 KPIs.
Several recent-month anomalies are planted so the detection engine
and domain agents have something real to catch:

  Finance (Jul):     Revenue drops 14% on weaker July demand
  Finance (Aug):     Gross margin compresses ~5pp as cost-of-goods
                     rises and mix shifts to lower-margin channels
  Sales   (Aug):     New-customer spike (paid acquisition burst)
  Sales   (Aug):     Return rate jumps on that cohort
  Ops     (Aug):     Fulfillment cost spike (expedited shipping)
  Ops     (Jul-Aug): On-time delivery dips (carrier capacity)

The June-July revenue/margin/on-time-delivery cluster is the kind of
'connected root cause' the synthesis agent should pick up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04",
          "2026-05", "2026-06", "2026-07", "2026-08"]


@dataclass
class KPI:
    id: str
    name: str
    domain: str
    unit: str
    direction: str        # 'higher_better' | 'lower_better'
    criticality: int      # 1..3
    values: List[float]
    targets: List[float]


# ---------- Finance -----------------------------------------------------

# Revenue ($M): mild growth trend through June, then -14% in July
# (demand softness), partial recovery in August (still below trend).
revenue_values = [4.20, 4.35, 4.55, 4.62, 4.70, 4.78, 4.10, 4.30]
revenue_targets = [4.30, 4.42, 4.55, 4.68, 4.82, 4.96, 5.10, 5.24]

# Gross margin (%): steady around 42-44%, then compression in August
# as COGS rises and mix shifts.
gross_margin_values = [43.5, 43.8, 44.0, 43.7, 44.1, 43.9, 43.6, 38.4]
gross_margin_targets = [43.0, 43.2, 43.4, 43.6, 43.8, 44.0, 44.2, 44.4]

# Cash on hand ($M): stable growth, slight dip in Aug from margin loss
cash_values = [11.5, 12.1, 12.8, 13.3, 13.9, 14.6, 14.9, 14.1]
cash_targets = [11.8, 12.4, 13.0, 13.6, 14.2, 14.8, 15.4, 16.0]


# ---------- Sales -------------------------------------------------------

# Units sold (k): steady growth then July dip, August partial rebound
units_sold_values = [82.0, 84.5, 86.1, 87.0, 88.3, 89.5, 78.2, 84.0]
units_sold_targets = [83.0, 85.0, 86.5, 88.0, 89.5, 91.0, 92.5, 94.0]

# New customers: stable, then a paid-acquisition burst in August
new_customers_values = [1240, 1280, 1310, 1340, 1380, 1410, 1450, 1980]
new_customers_targets = [1260, 1300, 1340, 1380, 1420, 1460, 1500, 1540]

# Return rate (%): typically 4.5-5.5%, jumps in August with the new cohort
return_rate_values = [4.8, 4.6, 5.0, 4.9, 5.1, 5.3, 5.0, 7.6]
return_rate_targets = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]


# ---------- Operations --------------------------------------------------

# Fulfillment cost ($k): steady, then spike in August (expedited shipping)
fulfillment_cost_values = [185, 188, 191, 194, 197, 201, 207, 248]
fulfillment_cost_targets = [190, 192, 194, 196, 198, 200, 202, 204]

# On-time delivery rate (%): very stable, dips July-August (carrier issues)
on_time_values = [96.4, 96.6, 96.2, 96.5, 96.3, 96.1, 94.1, 92.8]
on_time_targets = [96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0]


# ---------- KPI registry ------------------------------------------------

_ALL_KPIS: List[KPI] = [
    # Finance
    KPI("fin_revenue", "Revenue", "Finance", "$M", "higher_better", 3,
        revenue_values, revenue_targets),
    KPI("fin_gross_margin", "Gross Margin", "Finance", "%", "higher_better", 3,
        gross_margin_values, gross_margin_targets),
    KPI("fin_cash", "Cash on Hand", "Finance", "$M", "higher_better", 3,
        cash_values, cash_targets),
    # Sales
    KPI("sal_units_sold", "Units Sold", "Sales", "k units", "higher_better", 2,
        units_sold_values, units_sold_targets),
    KPI("sal_new_customers", "New Customers", "Sales", "count", "higher_better", 2,
        new_customers_values, new_customers_targets),
    KPI("sal_return_rate", "Return Rate", "Sales", "%", "lower_better", 2,
        return_rate_values, return_rate_targets),
    # Operations
    KPI("ops_fulfillment_cost", "Fulfillment Cost", "Operations", "$k",
        "lower_better", 2,
        fulfillment_cost_values, fulfillment_cost_targets),
    KPI("ops_on_time_delivery", "On-Time Delivery", "Operations", "%",
        "higher_better", 3,
        on_time_values, on_time_targets),
]


def load_all_kpis() -> List[KPI]:
    return list(_ALL_KPIS)


def kpis_by_domain() -> Dict[str, List[KPI]]:
    out: Dict[str, List[KPI]] = {}
    for k in _ALL_KPIS:
        out.setdefault(k.domain, []).append(k)
    return out


if __name__ == "__main__":
    for k in _ALL_KPIS:
        print(f"{k.domain:11s} | {k.name:20s} | "
              f"{' '.join(f'{v:6.2f}' for v in k.values)} "
              f"vs targets {' '.join(f'{t:6.2f}' for t in k.targets)}")