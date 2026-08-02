# Argus

> Multi-agent AI that spots trouble before it costs you.

Argus is an intelligent early warning system for organisational performance. Instead of waiting for last quarter's numbers to show a problem, Argus continuously scans KPI time series across Finance, Sales, and Operations, flags anything that drifts off trend or misses its target, and uses specialised AI agents to explain *why* it happened and *what to do about it*. The result is a single dashboard an executive can read in under a minute and walk away with three concrete actions.

This project was built as a submission for the **AI CAN DO IT** Tencent Cloud x UTM Hackathon, Case Study 2 (Intelligent Early Warning System for Organisational Performance), under the AI Agent Track, and developed using [WorkBuddy](https://www.codebuddy.ai) as the primary coding assistant.

## Live demo

> https://orchestra-ai-tl9o.onrender.com/

The app runs in demo mode out of the box, so the public link above is fully functional even without an LLM key. With a real key configured, the same URL serves live analyses.

## Why Argus

Most business dashboards are reactive. They show what already happened, and by the time a margin drop or a fulfillment cost spike is visible, the damage is done. Argus flips the model. It runs a statistical anomaly detector on every KPI every period, asks domain-expert agents to interpret the anomalies, and asks a chief-strategy-style synthesis agent to connect the dots across domains. The output is a ranked, prioritised, plain English story about what is about to hurt you, not what already did.

## Key features

### Anomaly detection engine
A pure-Python detector that scans every KPI in every month and flags anything that drifts more than 1.6 standard deviations from a 3-month rolling baseline or misses its target by more than 8 percent. Severity is direction-aware, so a margin drop and a margin spike are scored differently, and a fulfillment cost spike hurts more than a fulfillment cost dip.

### Three domain-expert agents
A dedicated agent for each business function. The Finance agent looks at revenue, margin, and cash. The Sales agent looks at units sold, new customers, return rate, pipeline, and channel mix. The Operations agent looks at fulfillment cost, on-time delivery, defect rate, and inventory. Each agent has a strict role boundary enforced by both the system prompt and the routing logic, so the Finance agent will never speculate on fulfillment issues.

### Cross-domain synthesis agent
After the three domain agents have weighed in, a synthesis agent reviews the top anomalies across all domains, identifies which ones share a common root cause, and produces an executive summary plus three prioritised actions. This is where a margin drop, a sales softness, and an operations cost spike get woven into a single narrative instead of three disconnected alerts.

### Interactive live dashboard
A single-page web app built with Chart.js. The dashboard shows three domain-level trend charts with actuals, targets, and anomaly months highlighted, plus an organisational health synthesis panel at the top and an expandable alert list at the bottom. Each chart supports click-to-expand into a larger view with pan and zoom, and every alert expands inline to show the full statistical context, root cause, and recommended action.

### Rate-limit-safe LLM integration
A thin OpenAI-compatible client that works with Groq, OpenAI, or any other provider that follows the chat-completions spec. Per-call throttling, exponential backoff on 429 and 5xx responses, and a soft retry for transient hiccups keep free-tier usage comfortably under rate limits. A deterministic built-in simulator runs when no key is configured, so the app is always demoable.

## Tech stack

- **Backend:** Python 3.12, Flask, Gunicorn
- **Frontend:** Vanilla JavaScript, Chart.js 4, chartjs-plugin-zoom
- **LLM:** Any OpenAI-compatible endpoint. Default is Groq with `openai/gpt-oss-20b`. OpenAI, DeepSeek, and other providers work by changing environment variables.
- **Data:** 8 months of synthetic KPIs across Finance, Sales, and Operations, with planted anomalies including a cross-domain cluster (July demand softness cascading into August margin compression and a fulfillment cost spike)

## How to run locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/argus.git
cd argus
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure the LLM (optional)

The app runs in demo mode with no configuration. To wire it to a real model, copy the example env file and fill in your key.

```bash
cp .env.example .env
```

Then edit `.env` with your provider of choice.

For Groq (the default, free-tier-friendly):

```
OPENAI_API_KEY=gsk_your_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=openai/gpt-oss-20b
```

For OpenAI:

```
OPENAI_API_KEY=sk_your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

Any OpenAI-compatible endpoint works. Set `OPENAI_BASE_URL` and `OPENAI_MODEL` accordingly.

### 4. Start the server

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

### 5. Verify

```bash
curl http://localhost:5000/api/health
```

The first load triggers the full pipeline. It loads the synthetic data, runs the anomaly detector, calls each domain agent on every flagged anomaly, and calls the synthesis agent for the executive summary. Subsequent loads are served from an in-memory cache. Click the **Refresh analysis** button in the top right to re-run the agents.

## Project structure

```
argus/
├── app.py                  Flask app and REST endpoints
├── anomaly_detector.py     Z-score and target-deviation detection
├── agents.py               Domain agents and synthesis agent
├── llm_client.py           OpenAI-compatible client and demo simulator
├── data/
│   ├── __init__.py
│   └── kpi_data.py         Synthetic 8-month KPI dataset
├── templates/
│   └── dashboard.html      Single-page dashboard markup
├── static/
│   ├── css/style.css       Light executive-dashboard theme
│   └── js/dashboard.js     Chart rendering, alert expansion, modal
├── requirements.txt        Python dependencies
├── Procfile                Production entry point for Render
├── runtime.txt             Pinned Python version
├── render.yaml             One-click Render Blueprint
└── .env.example            Configuration template
```

## Notes on the live demo

The deployed demo is a public URL on the Render free tier. The first request after a period of inactivity takes a few seconds because the service wakes from sleep. Once warm, the full analysis pipeline runs in roughly 30 seconds on a free-tier Groq key (8 anomalies plus 1 synthesis call, throttled to one call every 2.5 seconds).

## Acknowledgements

Built for the **AI CAN DO IT** Tencent Cloud x UTM Hackathon, Case Study 2 (Intelligent Early Warning System for Organisational Performance), under the AI Agent Track. Developed with [WorkBuddy](https://www.codebuddy.ai) as the primary coding assistant.

## License

This project is released for hackathon demonstration purposes. Feel free to fork, learn from, and adapt it.
