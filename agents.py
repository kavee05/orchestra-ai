"""
Domain expert agents and cross-domain synthesis agent.

Each domain agent has a STRICT role boundary - it only comments on its own
domain, and the system prompt explicitly forbids it from speculating on
other domains. This is enforced both by the system prompt and by routing
anomalies through `domain == agent.domain` checks.

When asked to analyze an anomaly the agent receives a compact prompt:
  - the anomaly essentials (kpi, month, value, target, deviation, z-score)
  - the recent 3-month history
  - the same-domain KPI snapshot for cross-KPI correlation

The agent returns a JSON object:
  {
    "root_cause": "<4-6 sentences>",
    "recommended_action": "<one concrete action>"
  }

The synthesis agent receives the TOP-N anomalies (by severity) across
domains with heavily-truncated root causes, and produces:
  {
    "executive_summary": "<3-4 sentences>",
    "prioritized_actions": ["...", "...", "..."]
  }

Rate-limit safety: per-alert calls are throttled by the LLMClient itself
(default 2.5s between real calls). Synthesis uses only the top 8 anomalies
so its prompt stays under ~1.5K input tokens even on busy runs.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Union

from llm_client import LLMClient, get_client
from anomaly_detector import Anomaly
from data.kpi_data import load_all_kpis


# ---------- logging ----------------------------------------------------
# All fallback paths log here so a live-demo failure leaves a paper trail
# in the terminal (stderr). Watch for "LLM FALLBACK" lines.
logger = logging.getLogger("kpi_ews.agents")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------- Domain system prompts (kept short to save tokens) -----------

DOMAIN_SYSTEM_PROMPTS = {
    "finance": (
        "You are a Senior Finance Analyst. You ONLY comment on finance "
        "KPIs (revenue, margin, cash, opex). Given an anomaly, return JSON "
        "with keys 'root_cause' (4-6 sentences, under 80 words) and "
        "'recommended_action' (one concrete action). Output ONLY the JSON."
    ),
    "sales": (
        "You are a Senior Commercial / Sales Ops leader. You ONLY comment "
        "on sales KPIs (units sold, new customers, return rate, pipeline, "
        "channel mix). Given an anomaly, return JSON with keys "
        "'root_cause' (4-6 sentences, under 80 words) and 'recommended_action' "
        "(one concrete action). Output ONLY the JSON."
    ),
    "operations": (
        "You are a Senior Operations leader (supply chain + fulfillment). "
        "You ONLY comment on operations KPIs (fulfillment cost, on-time "
        "delivery, defect rate, inventory). Given an anomaly, return JSON "
        "with keys 'root_cause' (4-6 sentences, under 80 words) and "
        "'recommended_action' (one concrete action). Output ONLY the JSON."
    ),
}

SYNTHESIS_SYSTEM_PROMPT = (
    "You are the Chief Strategy Officer reviewing KPI alerts across "
    "Finance, Sales, and Operations. Identify which flagged anomalies are "
    "CONNECTED (same root cause) vs independent. Respond with a single JSON "
    "object and nothing else - no prose, no markdown, no commentary. "
    "Schema: {\"executive_summary\": \"3-4 sentences of cross-domain "
    "narrative\", \"prioritized_actions\": [\"action 1\", \"action 2\", "
    "\"action 3\"]}. prioritized_actions must be exactly 3 strings ranked "
    "by urgency."
)


# ---------- Domain agent -----------------------------------------------

class DomainAgent:
    def __init__(self, domain: str, client: LLMClient):
        if domain not in DOMAIN_SYSTEM_PROMPTS:
            raise ValueError(f"unknown domain: {domain}")
        self.domain = domain
        self.client = client
        self.system_prompt = DOMAIN_SYSTEM_PROMPTS[domain]

    def _context_for(self, anomaly: Anomaly) -> str:
        same_domain = [k for k in load_all_kpis() if k.domain == self.domain]
        idx = anomaly.month_index
        snapshot = ", ".join(
            f"{k.name}={k.values[idx]}(tgt {k.targets[idx]})"
            for k in same_domain
        )
        history = ", ".join(
            f"{h['month']}:v{h['value']}/t{h['target']}"
            for h in anomaly.recent_history
        )
        return (
            f"DOMAIN: {self.domain}\n"
            f"KPI: {anomaly.kpi_name}\n"
            f"MONTH: {anomaly.month} | VALUE: {anomaly.value} {anomaly.unit} | "
            f"TARGET: {anomaly.target}\n"
            f"DEVIATION: {anomaly.deviation_pct}% | Z: {anomaly.z_score} | "
            f"SEVERITY: {anomaly.severity}\n"
            f"HISTORY: {history}\n"
            f"SAME-DOMAIN: {snapshot}"
        )

    def analyze(self, anomaly: Anomaly) -> Dict[str, Any]:
        assert anomaly.domain.lower() == self.domain
        user_prompt = (
            "Analyze this anomaly and respond as JSON.\n"
            + self._context_for(anomaly)
        )
        label = f"domain={self.domain} kpi={anomaly.kpi_name} month={anomaly.month}"
        raw = _chat_with_soft_retry(
            self.client,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            # 900 tokens gives openai/gpt-oss-20b (which often reasons out
            # loud before emitting JSON) enough headroom to finish a full
            # root_cause + recommended_action response AND close the JSON
            # braces/quote. 250 was causing the response to be truncated
            # mid-sentence, leaving us with un-parseable JSON.
            max_tokens=900,
            label=label,
        )
        # Surface error dicts from LLMClient.chat() unchanged.
        if isinstance(raw, dict) and raw.get("_error"):
            # Complete API failure (not a parse issue) — log at ERROR so
            # the actual exception type, HTTP status, and response body
            # show up clearly in Render logs. The user explicitly asked
            # to see the real error text, not just a generic placeholder,
            # so we can tell at a glance whether it was a timeout, a
            # 429, a 5xx, or an actual API error response. The `reason`
            # field already includes a 200-char body preview; we log it
            # as a single field to avoid duplication in the log line.
            logger.error(
                "LLM CALL FAILED (alert %s — %s/%s/%s): %s | model=%s",
                anomaly.alert_id, anomaly.domain, anomaly.kpi_name,
                anomaly.month,
                raw.get("reason", "unknown error"),
                raw.get("model", "?"),
            )
            return {
                "root_cause": (
                    f"[Live LLM unavailable: {raw.get('reason', 'error')}. "
                    f"Click 'Refresh analysis' to retry once rate limits reset.]"
                ),
                "recommended_action": "(retry to get an action)",
                "_error": True,
                "reason": raw.get("reason"),
            }
        parsed = _parse_json_response(raw, {
            "root_cause": "(analysis unavailable)",
            "recommended_action": "(no action proposed)",
        })
        if parsed.get("raw") is not None:
            # _parse_json_response only adds "raw" when it gave up on the
            # response (all parse paths failed). With openai/gpt-oss-20b
            # this happens whenever the model emits prose instead of strict
            # JSON. Instead of discarding the perfectly good analysis,
            # surface the raw text as the root_cause and trim if huge so
            # the alert card stays readable. The action line stays generic
            # because we can't reliably extract it from prose.
            raw_text = (parsed.get("raw") or "").strip()
            if raw_text:
                # Strip a leading "Root cause:" or similar label the model
                # sometimes prepends; the alert card already has its own
                # label.
                cleaned = re.sub(
                    r"^\s*(root\s*cause|analysis|answer|response)\s*[:\-]\s*",
                    "", raw_text, flags=re.I,
                ).strip()
                max_len = 800
                parsed["root_cause"] = (
                    cleaned[:max_len] + ("…" if len(cleaned) > max_len else "")
                )
                parsed["recommended_action"] = "See analysis above for details."
                logger.warning(
                    "LLM FALLBACK (alert %s — %s/%s/%s): JSON parse failed; "
                    "showing raw prose as root_cause (%d chars).",
                    anomaly.alert_id, anomaly.domain, anomaly.kpi_name,
                    anomaly.month, len(cleaned),
                )
            else:
                logger.warning(
                    "LLM FALLBACK (alert %s — %s/%s/%s): JSON parse failed "
                    "and raw response was empty; using placeholder.",
                    anomaly.alert_id, anomaly.domain, anomaly.kpi_name,
                    anomaly.month,
                )
        return parsed


# ---------- Synthesis agent --------------------------------------------

# Cap the number of anomalies sent to the synthesis LLM. Keeps the prompt
# under ~1.5K tokens even at the 200-char truncation we use below.
SYNTHESIS_MAX_ANOMALIES = 8
# How much of each per-alert root_cause to inline. Shorter = cheaper.
SYNTHESIS_CAUSE_TRUNC = 60


class SynthesisAgent:
    def __init__(self, client: LLMClient):
        self.client = client
        self.system_prompt = SYNTHESIS_SYSTEM_PROMPT

    def synthesize(self, anomalies: List[Anomaly],
                   per_domain_analyses: Dict[str, Dict[str, Any]]
                   ) -> Dict[str, Any]:
        # Sort by severity (High first) then severity_score and take top N.
        sev_rank = {"High": 3, "Medium": 2, "Low": 1}
        ranked = sorted(
            anomalies,
            key=lambda a: (sev_rank.get(a.severity, 0),
                           a.severity_score,
                           a.month_index),
            reverse=True,
        )
        top = ranked[:SYNTHESIS_MAX_ANOMALIES]

        lines = []
        for a in top:
            cause = per_domain_analyses.get(a.alert_id, {}).get(
                "root_cause", "")[:SYNTHESIS_CAUSE_TRUNC]
            lines.append(
                f"-[{a.severity}] {a.domain}/{a.kpi_name}/{a.month} "
                f"dev={a.deviation_pct}% z={a.z_score} | {cause}"
            )
        omitted = len(anomalies) - len(top)
        user_prompt = (
            f"FLAGGED: {len(anomalies)} total, top {len(top)} shown"
            f"{f' (+{omitted} lower-severity)' if omitted > 0 else ''}\n"
            + "\n".join(lines)
        )
        label = f"synthesis top={len(top)}/{len(anomalies)}"
        raw = _chat_with_soft_retry(
            self.client,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=400,
            label=label,
        )
        if isinstance(raw, dict) and raw.get("_error"):
            # Complete API failure (not a parse issue) — log at ERROR so
            # the actual exception type, HTTP status, and response body
            # show up clearly in Render logs (see DomainAgent.analyze).
            logger.error(
                "LLM CALL FAILED (synthesis — %d anomalies): %s | model=%s",
                len(anomalies), raw.get("reason", "unknown error"),
                raw.get("model", "?"),
            )
            return {
                "executive_summary": (
                    f"[Live LLM unavailable for synthesis: "
                    f"{raw.get('reason', 'error')}. The dashboard still shows "
                    f"per-alert analyses - click 'Refresh analysis' to retry.]"
                ),
                "prioritized_actions": [
                    "Retry the synthesis once the API rate limit resets.",
                    "Review the per-alert analyses below for individual actions.",
                    "Re-run analysis after a few minutes.",
                ],
                "_error": True,
                "reason": raw.get("reason"),
            }
        parsed = _parse_json_response(raw, {
            "executive_summary": "(synthesis unavailable)",
            "prioritized_actions": [
                "Review flagged anomalies.",
                "Assign owners per domain.",
                "Set follow-up review.",
            ],
        })
        if parsed.get("raw") is not None:
            # Same reasoning as DomainAgent: when the model returns prose
            # instead of JSON, surface the raw text as the executive
            # summary rather than discarding the real synthesis.
            raw_text = (parsed.get("raw") or "").strip()
            if raw_text:
                cleaned = re.sub(
                    r"^\s*(executive\s*summary|summary|answer|response)\s*[:\-]\s*",
                    "", raw_text, flags=re.I,
                ).strip()
                max_len = 1200  # synthesis is the cross-domain narrative,
                                # can run longer than per-alert prose
                parsed["executive_summary"] = (
                    cleaned[:max_len] + ("…" if len(cleaned) > max_len else "")
                )
                parsed["prioritized_actions"] = [
                    "Review the synthesis prose above for cross-domain actions.",
                    "Open individual alerts below for domain-specific actions.",
                    "Re-run analysis if a structured synthesis is needed.",
                ]
                logger.warning(
                    "LLM FALLBACK (synthesis — %d anomalies): JSON parse "
                    "failed; showing raw prose as executive_summary "
                    "(%d chars).",
                    len(anomalies), len(cleaned),
                )
            else:
                logger.warning(
                    "LLM FALLBACK (synthesis — %d anomalies): JSON parse "
                    "failed and raw response was empty; using placeholder.",
                    len(anomalies),
                )
        return parsed


# ---------- helpers ----------------------------------------------------

# Soft-retry config: after the LLMClient's own exponential-backoff chain
# gives up on a single alert, give it one more chance after a short pause.
# Saves a demo from a single transient 429 hiccup turning into a visibly
# broken alert card.
SOFT_RETRY_SECONDS = 4.0

# A model that occasionally reasons out loud (e.g. openai/gpt-oss-20b) will
# return text like:
#   "Reasoning: ...\n```json\n{...}\n```\nFinal answer:\n{... again ...}"
# The parser below tries progressively more aggressive extraction before
# giving up. The regex-extraction final step lets us still surface a usable
# synthesis even when JSON is malformed.


def _is_transient_error(reason: Optional[str]) -> bool:
    """True if an LLM error string indicates a rate-limit / timeout / 5xx
    that a single extra retry might recover from."""
    if not reason:
        return False
    r = reason.lower()
    return any(k in r for k in (
        "429", "rate_limit", "rate limit",
        "timeout", "timed out",
        "tpm", "tokens per",  # Groq per-minute token cap
        "503", "502", "500", "504",  # transient 5xx
        "connectionerror", "connection reset", "connection aborted",
    ))


def _chat_with_soft_retry(client: LLMClient, *,
                          messages: List[Dict[str, str]],
                          temperature: float,
                          max_tokens: int,
                          label: str) -> Union[str, Dict[str, Any]]:
    """Call client.chat() once. If it returns a transient error dict,
    sleep SOFT_RETRY_SECONDS and try once more before giving up."""
    raw = client.chat(messages=messages, temperature=temperature,
                      max_tokens=max_tokens)
    if isinstance(raw, dict) and raw.get("_error"):
        reason = raw.get("reason", "")
        if _is_transient_error(reason):
            logger.warning(
                "LLM TRANSIENT FAIL (%s): %s — soft-retrying after %.1fs",
                label, reason, SOFT_RETRY_SECONDS,
            )
            time.sleep(SOFT_RETRY_SECONDS)
            raw = client.chat(messages=messages, temperature=temperature,
                              max_tokens=max_tokens)
            if isinstance(raw, dict) and raw.get("_error"):
                logger.error(
                    "LLM CALL FAILED (soft-retry, %s): %s | model=%s",
                    label, raw.get("reason", ""), raw.get("model", "?"),
                )
            else:
                logger.info("LLM SOFT-RETRY OK (%s)", label)
    return raw

def _try_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort json.loads with small repairs (trailing commas, BOM,
    smart quotes). Returns a dict or None."""
    if not text:
        return None
    t = text.strip().lstrip("\ufeff")
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Repair: trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", t)
    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Repair: smart quotes -> straight quotes. Common when the LLM emits
    # typographic characters that are valid UTF-8 but break strict JSON
    # parsing on some Python versions / encodings.
    sq_swap = repaired.replace("\u201c", '"').replace("\u201d", '"') \
                       .replace("\u2018", "'").replace("\u2019", "'")
    if sq_swap != repaired:
        try:
            obj = json.loads(sq_swap)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    return None


def _find_balanced_json_objects(text: str) -> List[str]:
    """Return candidate {...} substrings with balanced braces, in source order.
    The model sometimes emits multiple JSON blocks (e.g. one inside a
    reasoning preamble and the final answer); we want the LAST one, which
    is usually the actual answer."""
    out: List[str] = []
    depth = 0
    in_str = False
    esc = False
    start = -1
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
                    start = -1
    return out


def _regex_extract_synthesis(text: str) -> Optional[Dict[str, Any]]:
    """Last-resort extractor. If we can find an 'executive_summary' value and
    a 'prioritized_actions' list in the prose, return them. Falls back to
    returning just the executive_summary if actions can't be located.
    Accepts both quoted ("key":) and unquoted (key:) keys since reasoning
    models often drop the quotes when they emit prose-form JSON.

    Like the per-alert regex path, the closing `"` is OPTIONAL — a response
    that was cut off mid-quote is still useful enough to surface with a
    trailing "…" so the user sees clean text instead of raw braces.
    """
    out: Dict[str, Any] = {}
    # Match either "executive_summary" or bare executive_summary, then
    # capture the first quoted value (with escape handling). 20 char min
    # keeps us from grabbing tiny fragments. The trailing `"?` lets us
    # match truncated responses that never closed the string.
    m = re.search(
        r'(?:"executive_summary"|executive_summary)\s*:\s*"((?:\\.|[^"\\])*)"?',
        text, re.DOTALL,
    )
    if m:
        val = m.group(1)
        truncated = (m.end() >= len(text.rstrip()))
        try:
            val = json.loads(f'"{val}"')
        except Exception:
            val = (val
                   .replace("\\n", " ")
                   .replace("\\\"", '"')
                   .replace("\\\\", "\\"))
        val = val.strip()
        if len(val) >= 20:
            if truncated:
                val = val.rstrip().rstrip(",").rstrip()
                if not val.endswith((".", "!", "?", "…")):
                    val += "…"
            out["executive_summary"] = val

    # Look for an array literal under prioritized_actions (quoted or not).
    # Same optional-closing treatment as above.
    m = re.search(
        r'(?:"prioritized_actions"|prioritized_actions)\s*:\s*\[((?:\\.|[^\[\]\\])*)\]?',
        text, re.DOTALL,
    )
    if m:
        try:
            arr = json.loads("[" + m.group(1) + "]")
            if isinstance(arr, list) and arr:
                out["prioritized_actions"] = [str(x) for x in arr][:3]
        except Exception:
            pass

    return out or None


def _parse_json_response(raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort JSON parse. Falls back gracefully if the model returns
    prose, a fenced JSON block, malformed JSON, or multiple JSON blocks."""
    if not raw:
        out = dict(fallback)
        out["raw"] = raw
        return out

    text = raw.strip()
    # Strip leading/trailing markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 1. Direct parse (after fence strip + small repair).
    parsed = _try_json_loads(text)
    if parsed:
        for k, v in fallback.items():
            parsed.setdefault(k, v)
        return parsed

    # 2. Walk balanced {...} objects; try the LAST one first since the
    # final answer is usually emitted after any reasoning preamble.
    candidates = _find_balanced_json_objects(text)
    for cand in reversed(candidates):
        parsed = _try_json_loads(cand)
        if parsed:
            for k, v in fallback.items():
                parsed.setdefault(k, v)
            return parsed

    # 3. Regex extraction as a true last resort.
    if "executive_summary" in fallback:
        extracted = _regex_extract_synthesis(text)
        if extracted:
            for k, v in fallback.items():
                extracted.setdefault(k, v)
            return extracted
    else:
        # Per-alert: try to fish out a root_cause / recommended_action
        # from the prose so the user still gets something useful.
        # The closing `"` is OPTIONAL because with openai/gpt-oss-20b we
        # sometimes see responses that get cut off mid-sentence (or
        # mid-quote) due to max_tokens. We detect truncation by checking
        # whether the text "looks like JSON" but the candidate value
        # extends to the end of the string — in that case we append "…"
        # to signal the cut-off to the reader.
        def _extract_quoted_field(pattern: str, src: str) -> Optional[tuple]:
            """Return (value, truncated_bool) or None if no match.
            truncated_bool=True means the field was found but never
            closed with a `"`, which usually means the model response
            was cut off mid-sentence."""
            m = re.search(pattern, src, re.DOTALL)
            if not m:
                return None
            raw_val = m.group(1)
            truncated = (m.end() >= len(src.rstrip()))
            try:
                val = json.loads(f'"{raw_val}"').strip()
            except Exception:
                # Unbalanced escapes etc. — best-effort: unescape common
                # sequences by hand so the user still sees clean text.
                val = (raw_val
                       .replace("\\n", " ")
                       .replace("\\\"", '"')
                       .replace("\\\\", "\\")
                       .strip())
            return val, truncated

        rc = _extract_quoted_field(
            r'"root_cause"\s*:\s*"((?:\\.|[^"\\])*)"?', text)
        ra = _extract_quoted_field(
            r'"recommended_action"\s*:\s*"((?:\\.|[^"\\])*)"?', text)
        # Check whether the whole text "looks like JSON that got cut off":
        # starts with '{' but never closes with a matching '}'.
        looks_truncated = (
            text.lstrip().startswith("{")
            and not _find_balanced_json_objects(text)
        )
        if rc or ra:
            out: Dict[str, Any] = {}
            if rc:
                val, tr = rc
                if tr or looks_truncated:
                    val = val.rstrip().rstrip(",").rstrip()
                    if not val.endswith((".", "!", "?", "…")):
                        val += "…"
                out["root_cause"] = val
            if ra:
                val, tr = ra
                if tr or looks_truncated:
                    val = val.rstrip().rstrip(",").rstrip()
                    if not val.endswith((".", "!", "?", "…")):
                        val += "…"
                out["recommended_action"] = val
            for k, v in fallback.items():
                out.setdefault(k, v)
            if looks_truncated:
                # Add a debug field so it's obvious in /api/analysis JSON
                # when a regex rescue was used (and the user knows the
                # response was truncated).
                out["_truncated"] = True
            return out

    # 4. Give up: return fallback (with raw text so it's debuggable).
    out = dict(fallback)
    out["raw"] = raw
    return out


# ---------- public entry point -----------------------------------------

def analyze_all(anomalies: List[Anomaly],
                client: Optional[LLMClient] = None
                ) -> Dict[str, Any]:
    """Run domain agents on every anomaly and then the synthesis agent.

    Returns:
      {
        "per_alert":   { alert_id: {root_cause, recommended_action, ...} },
        "synthesis":   { executive_summary, prioritized_actions },
        "model_info":  { real_api, model, base_url },
        "session_stats": { calls_attempted, calls_succeeded, ... },
        "has_errors":  bool - True if any LLM call failed and was surfaced.
      }
    """
    client = client or get_client()
    domain_agents = {d: DomainAgent(d, client) for d in DOMAIN_SYSTEM_PROMPTS}

    per_alert: Dict[str, Dict[str, Any]] = {}
    has_errors = False

    for a in anomalies:
        key = a.domain.lower()
        agent = domain_agents.get(key)
        if agent is None:
            continue
        analysis = agent.analyze(a)
        if analysis.get("_error"):
            has_errors = True

        # Attach metadata for the dashboard
        per_alert[a.alert_id] = {
            "alert_id": a.alert_id,
            "domain": a.domain,
            "kpi_name": a.kpi_name,
            "severity": a.severity,
            "month": a.month,
            "root_cause": analysis.get("root_cause", ""),
            "recommended_action": analysis.get("recommended_action", ""),
            "_error": analysis.get("_error", False),
            "reason": analysis.get("reason"),
        }

    # Synthesis (a single call; throttle handles the gap from the last
    # per-alert call automatically).
    synth = SynthesisAgent(client).synthesize(anomalies, per_alert)
    if synth.get("_error"):
        has_errors = True

    return {
        "per_alert": per_alert,
        "synthesis": synth,
        "model_info": {
            "real_api": client.use_real,
            "model": client.model,
            "base_url": client.base_url,
        },
        "session_stats": dict(client.session_stats),
        "has_errors": has_errors,
    }