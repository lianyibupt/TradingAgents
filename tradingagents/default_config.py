import os
from typing import Optional

from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
from tradingagents.llm_clients.model_catalog import get_model_options

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_TEMPERATURE":          "temperature",
}


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        config[key] = _coerce(raw, config.get(key))
    return config


def _is_usable_secret(value: Optional[str]) -> bool:
    if value is None:
        return False
    stripped = value.strip()
    return bool(stripped) and stripped != "placeholder"


def _default_model(provider: str, mode: str) -> str:
    # Auto-inferred providers should prefer the most broadly available,
    # stable model IDs instead of the newest frontier defaults. This path
    # exists to keep a fresh install runnable when the user has only set a
    # single provider key but not chosen models yet.
    auto_defaults = {
        "deepseek": {"quick": "deepseek-chat", "deep": "deepseek-chat"},
    }
    chosen = auto_defaults.get(provider, {}).get(mode)
    if chosen:
        return chosen

    for _, model in get_model_options(provider, mode):
        if model != "custom":
            return model
    raise ValueError(f"No default {mode} model configured for provider {provider!r}")


def _infer_provider_defaults(config: dict) -> dict:
    """Pick a usable provider/model set when the built-in OpenAI defaults can't run.

    This keeps fresh installs usable when the user has configured exactly one
    non-OpenAI provider key in ``.env`` but has not yet set
    ``TRADINGAGENTS_LLM_PROVIDER``. Explicit TRADINGAGENTS_* selections always win.
    """
    if os.environ.get("TRADINGAGENTS_LLM_PROVIDER"):
        return config

    current_provider = str(config.get("llm_provider", "openai")).lower()
    current_key_env = PROVIDER_API_KEY_ENV.get(current_provider)
    if current_key_env is None or _is_usable_secret(os.environ.get(current_key_env)):
        return config

    candidates = [
        provider
        for provider, env_var in PROVIDER_API_KEY_ENV.items()
        if provider != "azure" and env_var and _is_usable_secret(os.environ.get(env_var))
    ]
    if len(candidates) != 1:
        return config

    chosen = candidates[0]
    config["llm_provider"] = chosen
    if not os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM"):
        config["quick_think_llm"] = _default_model(chosen, "quick")
    if not os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"):
        config["deep_think_llm"] = _default_model(chosen, "deep")
    return config


DEFAULT_CONFIG = _infer_provider_defaults(_apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.5",
    "quick_think_llm": "gpt-5.4-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Sampling temperature, forwarded to every provider when set. None leaves
    # each provider at its own default. Lower values reduce run-to-run
    # variation on models that honor it; reasoning models largely ignore it
    # and no setting makes LLM output bit-identical across runs (see README).
    "temperature": None,
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal reasoning may still use English-friendly model behavior, but
    # user-facing reports default to Chinese in the local web experience.
    "output_language": "Chinese",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "analyst_concurrency_limit": 1,
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 20,             # max articles per ticker (ticker-news)
    "global_news_article_limit": 10,      # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # Search queries used by get_global_news for macro headlines. Extend or
    # replace to broaden geographic / sector coverage.
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "akshare",        # Options: alpha_vantage, yfinance, fmp, akshare
        "technical_indicators": "akshare",   # Options: alpha_vantage, yfinance, fmp, akshare
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance, fmp
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        "get_stock_data": "akshare,yfinance",  # akshare first, fallback to yfinance
    },
    # Benchmark for alpha calculation in the reflection layer.
    # ``benchmark_ticker`` (when set) overrides the suffix map for all
    # tickers; leave it None to use ``benchmark_map`` for auto-detection
    # based on the ticker's exchange suffix. SPY remains the US default
    # so the reflection label keeps reading "Alpha vs SPY" for US tickers
    # while non-US tickers get their regional index automatically.
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS":  "^NSEI",       # NSE India (Nifty 50)
        ".BO":  "^BSESN",      # BSE India (Sensex)
        ".T":   "^N225",       # Tokyo (Nikkei 225)
        ".HK":  "^HSI",        # Hong Kong (Hang Seng)
        ".L":   "^FTSE",       # London (FTSE 100)
        ".TO":  "^GSPTSE",     # Toronto (TSX Composite)
        ".AX":  "^AXJO",       # Australia (ASX 200)
        ".SS":  "000001.SS",   # Shanghai (SSE Composite)
        ".SZ":  "399001.SZ",   # Shenzhen (SZSE Component)
        "":     "SPY",         # default for US-listed tickers (no suffix)
    },
}))
