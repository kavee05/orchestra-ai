# KPI Early Warning System

> Hackathon prototype - **AI CAN DO IT** Tencent Cloud x UTM Hackathon  
> AI Agent Track - Case Study 2: *Intelligent Early Warning System for Organisational Performance*

A working Flask web app that **proactively** scans multi-domain KPI  
time-series data, flags anomalies with severity scores, asks  
domain-expert LLM agents to explain each one, and synthesizes the  
findings into an executive dashboard with prioritized recommended  
actions.

The intent: move organisations from reactive reporting ("we noticed  
margin dropped last quarter") to proactive intelligence ("here are the  
3 things likely to hurt us next month and what to do about each one").

---

## Features

1. **Web dashboard** (`/`)
   - Top: *Organizational Health Synthesis* - executive summary from  
     the synthesis agent plus three prioritized actions.
   - Middle: 3 trend charts (Finance / Sales / Operations) with  
     actuals (solid line), targets (dashed line), and anomaly months  
     marked in red.
   - Bottom: *Anomalies & Alerts* panel - every detected anomaly,  
     sorted High → Medium → Low, click-to-expand for full context.
2. **Real anomaly detection engine** (`anomaly_detector.py`)
   - 3-month rolling mean / stdev → **z-score**.
   - Per-month **target deviation %**.
   - Direction-aware severity weighting (a margin *drop* and a margin  
     *spike* are not the same).
   - Tunable thresholds (defaults: `|z| ≥ 1.6` or `|Δ target| ≥ 8%`).
3. **Domain expert agents** (`agents.py`)
   - **Finance Agent**, **Sales Agent**, **Operations Agent** - each  
     with a strict role boundary enforced both by the system prompt  
     and by routing (anomalies only flow to their own domain agent).
   - Receives the anomaly + 3-month history + same-domain KPI snapshot.
   - Outputs JSON `{root_cause, recommended_action}` (4-6 sentences
     - 1 concrete action).
4. **Synthesis agent**
   - Reviews all flagged anomalies across domains.
   - Detects cross-domain connections (e.g. a sales drop and a margin  
     drop from the same root cause).
   - Outputs `{executive_summary, prioritized_actions[3]}`.
5. **OpenAI-compatible LLM integration** (`llm_client.py`)
   - Set `OPENAI_API_KEY` + `OPENAI_BASE_URL` + `OPENAI_MODEL` and it  
     works with Groq, OpenAI, Tencent 混元, DeepSeek, or any other  
     provider that follows the `/v1/chat/completions` spec.
   - **Demo / fallback mode** - if no key is set (or the key starts  
     with `demo`), a deterministic built-in simulator produces  
     realistic analyst-style responses so the app always runs end-to-  
     end for a live demo.
6. **Synthetic data** (`data/kpi_data.py`)
   - 8 months × 8 KPIs across Finance (revenue, gross margin, cash),  
     Sales (units sold, new customers, return rate), Operations  
     (fulfillment cost, on-time delivery).
   - 6 planted anomalies in the most recent months including a  
     cross-domain root cause cluster (Jul demand softness → margin  
     compression → ops cost spike).

---

## File structure

```
kpi-ews/
├── app.py                  Flask app + REST endpoints
├── anomaly_detector.py     Real z-score + target-deviation detection
├── agents.py               Domain agents + synthesis agent
├── llm_client.py           OpenAI-compatible client + demo simulator
├── data/
│   ├── __init__.py
│   └── kpi_data.py         Synthetic 8-month KPI dataset
├── templates/
│   └── dashboard.html      Single-page dashboard markup
├── static/
│   ├── css/style.css       Light executive-dashboard theme
│   └── js/dashboard.js     Chart rendering, alert expansion, refresh
├── requirements.txt        flask + requests
├── .env.example            Copy to `.env` and fill in for live LLM
└── README.md               ← this file
```

---

## How to run

### 1. Install dependencies

```bash
cd kpi-ews
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure the LLM (optional)

The app runs in **demo mode** out of the box. To use a real LLM:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY (and optionally BASE_URL/MODEL)
```

**Groq** (default settings):

```
OPENAI_API_KEY=gsk_...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.1-70b-versatile
```

**OpenAI**:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

The app reads env vars on startup, so you can also set them in your  
shell instead of using a `.env` file.

### 3. Run the server

```bash
python app.py
```

Open **<http://localhost:5000>** in your browser.

### 4. Verify

```bash
curl http://localhost:5000/api/health
# -> {"status":"ok","model":"llama-3.1-70b-versatile","real_api":false,...}
```

The first GET to `/api/bootstrap` will:

1. Load the synthetic KPI dataset.
2. Run the anomaly detector (z-score + target deviation).
3. Call each domain agent (Finance / Sales / Operations) for every  
   flagged anomaly.
4. Call the synthesis agent to produce the executive summary.
5. Cache everything in-memory; subsequent loads are instant.

Click `⟳ Refresh analysis` to re-run all agents (useful after changing  
your API key).

---

## Architecture / data flow

```
Synthetic data ──► anomaly_detector ──► list of Anomaly objects
                          │
                          ▼
                   agents.DomainAgent
                   (Finance / Sales / Ops)
                          │  (parallel fan-out, JSON in/out)
                          ▼
                  {root_cause, recommended_action}
                          │
                          ▼
                  agents.SynthesisAgent
                          │
                          ▼
                  {executive_summary, prioritized_actions[3]}
                          │
                          ▼
                  Flask app.py → /api/bootstrap
                          │
                          ▼
              templates/dashboard.html + Chart.js
```

---

## Anomaly scoring (for the curious)

For each KPI value in each month:

```
z_score      = (value - rolling_mean_3m) / rolling_std_3m
dev_pct      = (value - target) / target * 100
flag         = |z_score| >= 1.6  OR  |dev_pct| >= 8
score        = max(|z_score|, |dev_pct|/10) * criticality_weight
severity     = score >= 7 ? High : score >= 4 ? Medium : Low
```

Criticality weights: Finance-3 / Ops-3 = 3, Sales = 2. Lower-better  
KPIs (cost, return rate) flip the sign so an upward blip is what hurts  
the score.

---

## Demo walkthrough (for the judges)

1. Open the page - the synthesis panel summarises the cross-domain  
   story, the three charts show where things are tracking, and the  
   alerts panel flags what's broken.
2. The first thing they'll notice: a **High** severity *Gross Margin*  
   alert in **Finance** for August.
3. Click it: it expands to show (a) the z-score and target-deviation  
   that triggered it, (b) the 3-month history table, (c) the Finance  
   Agent's root cause (margin compression + cost-of-goods/mix shift)  
   and a single concrete action.
4. Scroll to the *Sales* alert for *Return Rate* (August) - notice it  
   appears connected to the *New Customers* spike (August).
5. The synthesis panel at the top should call this out as a separate  
   connected cluster from the Finance/Ops margin issue.

---

## Deployment (Render.com - free tier)

Quickest path to a public URL:

1. Push this repo to GitHub (don't commit `.env` - it has your Groq key).
2. On render.com: New + → Web Service → connect the GitHub repo.
3. Render auto-detects Python from `runtime.txt` (3.12.7) and runs the
   `Procfile` (`gunicorn app:app`).
4. Add env vars in the Render dashboard: `OPENAI_API_KEY`,
   `OPENAI_BASE_URL=https://api.groq.com/openai/v1`,
   `OPENAI_MODEL=openai/gpt-oss-20b`, `LLM_MIN_INTERVAL=2.5`.
5. Deploy → wait ~3 min → live URL like `https://kpi-ews.onrender.com`.

Free tier sleeps after 15 min idle; wakes on first request. Fine for demos.
See `render.yaml` for a one-click Blueprint deploy alternative.

---

## Limitations / next steps (out of scope for this prototype)

- Single-tenant; no auth.
- Synthetic data is hard-coded; swap `data/kpi_data.py` for a CSV or  
  SQL pull to use real numbers.
- Severity thresholds tuned for the demo dataset; expose as env vars  
  for production tuning.
- No persistence; cache resets on restart.
- LLM latency: on first load the agents run sequentially. Add a  
  thread pool or move to streaming responses for production.

---

Built under a hackathon deadline. Functional completeness > polish.
