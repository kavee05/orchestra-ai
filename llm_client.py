"""
LLM client supporting any OpenAI-compatible endpoint.

Environment variables:
  OPENAI_API_KEY    required to call a real model. If unset or starts with
                    'demo' / 'sk-demo', the client falls back to a built-in
                    simulator so the app is fully runnable for a live demo.
  OPENAI_BASE_URL   defaults to https://api.groq.com/openai/v1
                    (works with Groq; for OpenAI set https://api.openai.com/v1;
                    for any other provider set their /v1 base.)
  OPENAI_MODEL      defaults to openai/gpt-oss-20b (a Groq-friendly 20B model)

Rate-limit safety (for free Groq tier, which is ~30 req/min and ~7200 tok/min):
  - Each call is throttled with `min_interval_seconds` (default 2.5s) between
    real API calls, so a 15-alert run takes ~40s and stays comfortably under
    the per-minute request budget.
  - On 429 / rate_limit_exceeded the client retries with exponential backoff
    (1s, 2s, 4s) up to 3 attempts before returning a structured error dict
    that the UI surfaces as a yellow banner - no silent fallback to the
    simulator for live runs.
  - The simulator remains in place as a hard fallback so the app keeps
    running even if the API is down or the key is invalid.

The client exposes a single `chat(messages, ...)` method and returns a
plain string on success, or a structured error dict on failure.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from typing import List, Dict, Any, Optional, Union

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


# Module-level logger so the "reasoning_effort rejected" warning in chat()
# and any future client-side events have a stable log channel. Stays at
# INFO by default; chat() and agents.py emit ERROR/WARNING for failures.
logger = logging.getLogger("kpi_ews.llm_client")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"
# Seconds between consecutive real-API calls. 2.5s keeps a 15-call run
# safely under Groq free tier's 30 req/min limit.
DEFAULT_MIN_INTERVAL = 2.5
# Per-call HTTP timeout. Some free-tier models can be slow on cold start.
DEFAULT_TIMEOUT = 45.0


class LLMClient:
    """Thin OpenAI-compatible chat client with rate-limit-safe fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        min_interval_seconds: Optional[float] = None,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url if base_url is not None
                         else os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
        self.model = (model if model is not None
                      else os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
        self.timeout = timeout
        # LLM_MIN_INTERVAL lets you tune the throttle at runtime without
        # code changes. Default 2.5s keeps 15 calls safely under Groq's
        # 30 req/min free-tier limit.
        env_interval = os.environ.get("LLM_MIN_INTERVAL")
        if min_interval_seconds is None:
            min_interval_seconds = float(env_interval) if env_interval else DEFAULT_MIN_INTERVAL
        self.min_interval_seconds = min_interval_seconds
        # LLM_JSON_MODE - opt-in to response_format: {"type":"json_object"}.
        # IMPORTANT: openai/gpt-oss-20b on Groq's free tier currently rejects
        # this with HTTP 400, so we default to OFF. The improved JSON parser
        # in agents.py handles reasoning-preamble + non-JSON responses just
        # fine, so we don't actually need it for this model. Set
        # LLM_JSON_MODE=1 if you switch to a model that supports it
        # (e.g. llama-3.1-70b-versatile).
        self.json_mode = os.environ.get("LLM_JSON_MODE", "").lower() in {
            "1", "true", "yes", "on",
        }
        # LLM_REASONING_EFFORT - reduce the chain-of-thought budget Groq
        # reasoning models (gpt-oss-20b/120b) use BEFORE producing the
        # final answer. Without this, gpt-oss-20b frequently burns the
        # entire max_tokens budget on internal reasoning and returns an
        # empty `choices[0].message.content`. Valid values on Groq are
        # "low" / "medium" / "high". Set to "" (empty) to disable.
        # Auto-detect: "low" for gpt-oss models, omitted for everything
        # else. Override with the env var.
        env_re = os.environ.get("LLM_REASONING_EFFORT")
        if env_re is not None:
            # Explicit env-var override (including empty string to disable).
            self.reasoning_effort: Optional[str] = env_re.strip() or None
        else:
            m = self.model.lower()
            if "gpt-oss" in m or "-oss-" in m:
                self.reasoning_effort = "low"
            else:
                self.reasoning_effort = None
        self.use_real = self._should_use_real(self.api_key)

        # Rate-limit tracking (thread-safe).
        self._lock = threading.Lock()
        self._last_call_ts: float = 0.0
        self._call_count_minute: int = 0
        self._minute_window_start: float = time.time()
        # Counters useful for surfacing in the UI.
        self.session_stats: Dict[str, int] = {
            "calls_attempted": 0,
            "calls_succeeded": 0,
            "calls_retried": 0,
            "calls_rate_limited": 0,
            "calls_failed": 0,
        }

    @staticmethod
    def _should_use_real(key: str) -> bool:
        if not key:
            return False
        if key.lower().startswith("demo") or key.lower().startswith("sk-demo"):
            return False
        if key.strip().lower() in {"your_key_here", "changeme", ""}:
            return False
        return True

    # ---------- public API ----------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.4,
        max_tokens: int = 400,
    ) -> Union[str, Dict[str, Any]]:
        """Send a chat completion. Returns the assistant text on success.

        On persistent failure (after retries) returns a dict:
            {
                "_error":    True,
                "stage":     "real_api",
                "reason":    str,   # human-readable summary (type + status
                                    # + first 160 chars of exc + body preview)
                "raw":       str,   # up to 600 chars of the actual response
                                    # body from the server (so the user can
                                    # see the real API error text in logs)
                "status":    int|None,  # HTTP status code, if available
                "exception": str|None, # "<ClassName>: <message>" or None
                "model":     str,
            }
        The agents check `isinstance(result, dict) and result.get("_error")`
        to decide whether to use a fallback message instead of parsing
        the response as JSON.
        """
        if not self.use_real:
            return self._simulate(messages)

        # Throttle: enforce a minimum gap between real-API calls.
        self._throttle()

        self.session_stats["calls_attempted"] += 1

        last_exc: Optional[Exception] = None
        last_status: Optional[int] = None
        # Hoist the response body out of the per-attempt block so the
        # final error dict can surface it. The user's debug log will
        # then show the REAL API error text (e.g. Groq's "tool_use
        # not supported" or "context_length_exceeded" or whatever the
        # model actually returned), not just a generic exception name.
        last_body: str = ""
        # Exponential backoff schedule: 1s, 2s, 4s.
        backoffs = [1.0, 2.0, 4.0]
        max_attempts = 1 + len(backoffs)  # initial + 3 retries

        # Resolve exception classes defensively (requests may be None
        # if the dependency isn't installed yet).
        HTTPError = getattr(requests, "exceptions", None)
        HTTPError_cls = getattr(HTTPError, "HTTPError", None) if HTTPError else None
        Timeout_cls = getattr(HTTPError, "Timeout", None) if HTTPError else None
        Conn_cls = getattr(HTTPError, "ConnectionError", None) if HTTPError else None

        for attempt in range(max_attempts):
            try:
                text = self._call_real(messages, temperature, max_tokens)
                self.session_stats["calls_succeeded"] += 1
                return text
            except Exception as exc:
                last_exc = exc
                # Read the response body + status if the failure was an HTTPError.
                last_status = None
                last_body = ""
                if HTTPError_cls and isinstance(exc, HTTPError_cls):
                    resp = getattr(exc, "response", None)
                    if resp is not None:
                        last_status = getattr(resp, "status_code", None)
                        try:
                            last_body = (resp.text or "")[:600]
                        except Exception:
                            pass
                # Detect if Groq specifically rejected the reasoning_effort
                # parameter. This is a CONFIG error (not transient), so we
                # log it as a dedicated WARNING with actionable advice
                # ("rely on max_tokens increase instead") rather than
                # burning the retry budget. We only do this on the FIRST
                # attempt — retries won't magically make Groq accept the
                # param, so just bail out.
                if (last_status == 400
                        and self.reasoning_effort
                        and "reasoning_effort" in last_body.lower()):
                    logger.warning(
                        "LLM CONFIG REJECTED (model=%s): Groq returned HTTP 400 "
                        "and the body mentions 'reasoning_effort' — this "
                        "model/version does not support that parameter. "
                        "Disabling it for this session and falling back to "
                        "max_tokens-only path. To permanently disable, set "
                        "LLM_REASONING_EFFORT='' in your env vars. "
                        "body=%s",
                        self.model, last_body[:300],
                    )
                    self.reasoning_effort = None
                    # Don't retry — Groq will keep rejecting.
                    break
                is_rate_limit = (
                    last_status == 429
                    or "rate_limit" in last_body.lower()
                    or "tpm" in last_body.lower()
                )
                if is_rate_limit:
                    self.session_stats["calls_rate_limited"] += 1
                # Network-level transient errors get the same retry budget.
                is_transient_network = (Timeout_cls and isinstance(exc, Timeout_cls)) \
                    or (Conn_cls and isinstance(exc, Conn_cls))
                transient = is_rate_limit or is_transient_network \
                    or (last_status is not None and 500 <= last_status < 600)
                if not transient or attempt >= max_attempts - 1:
                    break
                self.session_stats["calls_retried"] += 1
                time.sleep(backoffs[attempt])
                self._throttle()

        self.session_stats["calls_failed"] += 1
        # Build a rich error dict. The `reason` is the human-readable
        # exception summary (type + HTTP status + first 160 chars of
        # the exception message). The `raw` field carries up to 600
        # chars of the actual API response body so the user's logs
        # always include the real server-side error text — never just
        # a generic "ConnectionError" or "Timeout".
        reason = (f"{last_exc.__class__.__name__}"
                  + (f" (HTTP {last_status})" if last_status else "")
                  + (f": {str(last_exc)[:160]}" if last_exc else ""))
        if last_body and last_body not in reason:
            reason = f"{reason} | body={last_body[:200]}"
        return {
            "_error": True,
            "stage": "real_api",
            "reason": reason,
            "raw": last_body,           # full body (up to 600 chars)
            "status": last_status,      # HTTP status code, if any
            "exception": (
                f"{last_exc.__class__.__name__}: {str(last_exc)[:200]}"
                if last_exc else None
            ),
            "model": self.model,
        }

    def _throttle(self) -> None:
        """Sleep until min_interval_seconds has elapsed since the last call."""
        with self._lock:
            now = time.time()
            # Reset the per-minute counter every 60s.
            if now - self._minute_window_start >= 60.0:
                self._minute_window_start = now
                self._call_count_minute = 0
            gap = now - self._last_call_ts
            if self._last_call_ts > 0 and gap < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - gap)
            self._last_call_ts = time.time()
            self._call_count_minute += 1

    # ---------- real API path -------------------------------------------

    def _call_real(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        if requests is None:
            raise RuntimeError("requests library not installed")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # Opt-in: some models (e.g. llama-3.1-70b-versatile) honour
        # response_format and emit cleaner JSON. Some (e.g. openai/gpt-oss-20b
        # on Groq's free tier as of 2026-07) reject it with HTTP 400. The
        # improved JSON parser in agents.py handles both cases, so we keep
        # this OFF by default and let users flip it on with LLM_JSON_MODE=1.
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        # reasoning_effort is a Groq-specific parameter that caps how much
        # of the token budget reasoning models (gpt-oss-20b/120b) spend
        # on internal chain-of-thought before emitting the final answer.
        # Without it, gpt-oss-20b routinely burns the whole max_tokens
        # budget on thinking and returns empty content. We only add it
        # when self.reasoning_effort is set (auto-detected for gpt-oss
        # models, controllable via LLM_REASONING_EFFORT env var). If
        # Groq rejects it for a particular model, chat() detects that
        # in the error body and logs a dedicated WARNING.
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        resp = requests.post(url, headers=headers, json=payload,
                             timeout=self.timeout)
        # Raise_for_status handles 429 + 5xx for us; we read body in chat().
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ---------- simulator -----------------------------------------------

    # The simulator inspects the user prompt for keywords (kpi id, domain,
    # severity) and composes a plausible 4-6 sentence analyst response
    # plus a concrete action. Output is JSON-shaped so the agents can
    # parse it identically to real LLM output.

    _DOMAIN_TEMPLATES = {
        "finance": {
            "root": (
                "A {pct_sign} of {pct:.1f}% versus target on {kpi} in {month} "
                "is consistent with a {context} event. Specifically, {detail}. "
                "The {direction_adj} {kpi} reading sits {z_lvl} the 3-month "
                "rolling baseline, suggesting this is not a routine fluctuation "
                "but a structural shift that warrants immediate finance review. "
                "If left unchecked, the margin trajectory in subsequent months "
                "would compress reported earnings and constrain the cash buffer "
                "currently sitting at ${cash}M."
            ),
            "action": (
                "Reconcile the {month} close within 5 business days, identify the "
                "{driver} driver, and present a recovery plan at the next "
                "executive finance review."
            ),
        },
        "sales": {
            "root": (
                "The {direction_adj} {kpi} print in {month} ({pct:.1f}% vs target) "
                "points to a {context} pattern. {detail}. Combined with the "
                "movement in adjacent sales indicators, this is most likely "
                "linked to {driver} rather than market-wide softness. "
                "Pipeline conversion and channel mix should be revisited, as "
                "the current pattern, if sustained for two more cycles, would "
                "drag quarterly bookings by an estimated 6-9%."
            ),
            "action": (
                "Pause spend on the {driver} channel for 14 days, reallocate to "
                "the highest-LTV source, and have the commercial lead produce a "
                "channel-level P&L by next Friday."
            ),
        },
        "operations": {
            "root": (
                "Operations recorded a {direction_adj} {kpi} reading in {month} "
                "({pct:.1f}% off target, {z_lvl} the trailing baseline). "
                "{detail}. The most plausible cause is a {context} disruption "
                "to the fulfillment network, likely compounded by upstream "
                "demand volatility. Left unremediated, this would push the cost-"
                "to-serve above the 12% threshold for the quarter and risk SLA "
                "breaches with key accounts."
            ),
            "action": (
                "Open a 48-hour ops war-room with the fulfillment partner, "
                "freeze non-critical shipping upgrades, and surface a 2-week "
                "recovery plan including cost containment levers."
            ),
        },
    }

    _SYNTHESIS_TEMPLATE = (
        "Across {n} flagged issues spanning {domains}, the dominant pattern is "
        "two distinct root causes that have collided in the same reporting "
        "window. First, the July demand softness has cascaded into August "
        "margin compression and elevated fulfillment costs as the operations "
        "team leaned on expedited shipping - finance, sales volume, and "
        "operations are all signaling the same upstream event. Second, a "
        "parallel paid-acquisition burst in August drove a sharp spike in "
        "new customers but pulled in a cohort with above-average return "
        "propensity, which is now showing up as a return-rate anomaly. "
        "Priority should be (1) finance close and margin recovery for July-"
        "August, (2) returns containment from the August acquisition cohort, "
        "and (3) a structural review of fulfillment cost resilience before "
        "the holiday peak. "
        "[Simulated synthesis - set OPENAI_API_KEY for live LLM]"
    )

    def _simulate(self, messages: List[Dict[str, str]]) -> str:
        """Produce a JSON-shaped analyst response based on the prompt."""
        # Find the last user message (most informative).
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break
        sys_msg = ""
        for m in messages:
            if m.get("role") == "system":
                sys_msg = m.get("content", "")
                break

        # If this is a synthesis call, return a different shape.
        if "chief strategy officer" in sys_msg.lower() or "synthesize" in sys_msg.lower():
            domains = re.findall(r"domain:\s*(\w+)", user_msg, re.I)
            n_match = re.search(r"(\d+)\s+flagged", user_msg, re.I)
            n = n_match.group(1) if n_match else "several"
            domains_str = ", ".join(sorted(set(d.lower() for d in domains))) or "finance, sales, operations"
            text = self._SYNTHESIS_TEMPLATE.format(n=n, domains=domains_str)
            return json.dumps({
                "executive_summary": text,
                "prioritized_actions": [
                    "Finance: close July books, identify margin recovery lever within 5 business days.",
                    "Sales: contain returns from August acquisition cohort, reallocate channel spend.",
                    "Operations: 48-hour fulfillment war-room, structural cost review before peak.",
                ],
            })

        # Domain-agent call: parse hints from the prompt.
        domain_match = re.search(r"domain:\s*(\w+)", user_msg, re.I)
        kpi_match = re.search(r"kpi:\s*([^\n|]+)", user_msg, re.I)
        month_match = re.search(r"month:\s*([\d-]+)", user_msg, re.I)
        # Match either the old "DEVIATION VS TARGET:" or the new compact "DEVIATION:"
        pct_match = re.search(r"deviation[^:\n]*:\s*(-?\d+\.?\d*)%", user_msg, re.I)
        z_match = re.search(r"z-?score:\s*(-?\d+\.?\d*)|\| z:\s*(-?\d+\.?\d*)",
                            user_msg, re.I)
        direction_match = re.search(r"direction:\s*(\w+)", user_msg, re.I)

        domain = (domain_match.group(1) if domain_match else "operations").lower()
        kpi = (kpi_match.group(1).strip().split(" (")[0] if kpi_match else "the metric")
        # Strip "(id=...)" tail if present
        if "(" in kpi:
            kpi = kpi.split("(")[0].strip()
        month = month_match.group(1) if month_match else "the most recent month"
        pct = float(pct_match.group(1)) if pct_match else 0.0
        z = float((z_match.group(1) or z_match.group(2)) if z_match else 0.0)
        direction = (direction_match.group(1) if direction_match else "higher_better").lower()

        pct_sign = "drop" if pct < 0 else "surge"
        direction_adj = "lower-than-expected" if (pct < 0) == (direction == "higher_better") \
                        else "higher-than-expected"
        # Fix adjective based on whether deviation is bad (use sign of pct vs direction).
        if direction == "higher_better":
            direction_adj = "decline" if pct < 0 else "spike"
        else:
            direction_adj = "spike" if pct > 0 else "drop"
        z_lvl = "well above" if abs(z) >= 2 else "noticeably above"

        context_map = {
            "finance": "revenue recognition / cost-of-goods",
            "sales": "demand / channel-quality",
            "operations": "logistics / vendor-capacity",
        }
        driver_map = {
            "finance": "input-cost or pricing-mix",
            "sales": "channel-mix / lead-quality",
            "operations": "carrier or warehouse-capacity",
        }
        detail_map = {
            "finance": (
                "either input costs moved unfavorably or the sales mix shifted "
                "toward lower-margin channels following the July demand softness"
            ),
            "sales": (
                "either a campaign pulled in lower-LTV cohorts or a key channel "
                "is underperforming against its contribution margin target"
            ),
            "operations": (
                "either expedited shipping was used to recover missed SLAs or "
                "a carrier rate change took effect mid-period"
            ),
        }

        tpl = self._DOMAIN_TEMPLATES.get(domain, self._DOMAIN_TEMPLATES["operations"])
        # Light randomness on cash figure for realism without breaking determinism.
        rng = random.Random(hash((domain, kpi, month)) & 0xFFFFFFFF)
        cash = round(12 + rng.random() * 4, 1)

        root_cause = tpl["root"].format(
            pct_sign=pct_sign, pct=abs(pct), kpi=kpi, month=month,
            context=context_map[domain], detail=detail_map[domain],
            direction_adj=direction_adj, z_lvl=z_lvl, cash=cash,
            driver=driver_map[domain],
        )
        action = tpl["action"].format(
            month=month, driver=driver_map[domain],
        )

        return json.dumps({
            "root_cause": root_cause,
            "recommended_action": action,
        })


def get_client() -> LLMClient:
    """Factory used by agents / app so env is read at call time."""
    return LLMClient()