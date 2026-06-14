# -*- coding: utf-8 -*-
"""TradingAgents Web Interface - Multi-Agent LLM Trading Analysis Dashboard"""

import os
import sys
import json
import locale
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import deque
import sqlite3
import uuid
import re

from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from web.services.workspace import (
    build_default_workspace,
    normalize_workspace_payload,
    recalculate_account_state,
    refresh_position_after_analysis,
    PRICE_SOURCE_ANALYSIS_REFRESH,
)
from web.services.account_analysis import (
    save_account_analysis_artifacts,
    list_account_analysis_history,
)
from web.services.database import get_database_manager

# Load .env
load_dotenv()

# Ensure UTF-8
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        pass

# --- Project paths ---
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_web_dir = os.path.dirname(os.path.abspath(__file__))
_template_dir = os.path.join(_web_dir, 'templates')
_static_dir = os.path.join(_web_dir, 'static')

app = Flask(
    __name__,
    template_folder=_template_dir,
    static_folder=_static_dir,
    static_url_path='/static',
)


# ---------------------------------------------------------------------------
# SQLite history database
# ---------------------------------------------------------------------------

_DB_PATH = os.path.join(_project_root, 'web', 'history.db')


def _init_db():
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                decision TEXT,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_history_ticker ON analysis_history(ticker)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_history_created ON analysis_history(created_at DESC)
        ''')


_init_db()
db_manager = get_database_manager()


def _save_to_history(ticker: str, date_str: str, decision: str, result_data: dict):
    """Save an analysis result to history."""
    record_id = str(uuid.uuid4())[:8]
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(
            'INSERT OR REPLACE INTO analysis_history (id, ticker, date, decision, created_at, data) VALUES (?, ?, ?, ?, ?, ?)',
            (record_id, ticker, date_str, decision, created_at, json.dumps(result_data, ensure_ascii=False))
        )
    return record_id


def _get_history(limit: int = 50, offset: int = 0):
    """Get analysis history list."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT id, ticker, date, decision, created_at FROM analysis_history ORDER BY created_at DESC LIMIT ? OFFSET ?',
            (limit, offset)
        ).fetchall()
        total = conn.execute('SELECT COUNT(*) FROM analysis_history').fetchone()[0]
    return [dict(r) for r in rows], total


def _get_history_detail(record_id: str):
    """Get full detail of a history record."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM analysis_history WHERE id = ?', (record_id,)
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result['data'] = json.loads(result['data'])
    return result


def _delete_history(record_id: str):
    """Delete a history record."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute('DELETE FROM analysis_history WHERE id = ?', (record_id,))


def _clear_history():
    """Clear all history."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute('DELETE FROM analysis_history')


# ---------------------------------------------------------------------------
# Progress callback handler
# ---------------------------------------------------------------------------
class WebProgressHandler(BaseCallbackHandler):
    """LangChain callback that feeds progress updates to the web UI."""

    def __init__(self, progress_list: list):
        super().__init__()
        self._progress = progress_list
        self._lock = threading.Lock()
        self._llm_count = 0
        self._tool_count = 0
        self._tokens_in = 0
        self._tokens_out = 0
        self._last_report = ""

    def _log(self, msg: str):
        with self._lock:
            self._progress.append(msg)
            self._last_report = msg

    def on_chat_model_start(self, serialized, messages, **kwargs):
        with self._lock:
            self._llm_count += 1
            c = self._llm_count
        # Only log every 5th LLM call to avoid spam
        if c <= 3 or c % 5 == 0:
            self._log(f"🤖 LLM call #{c}...")

    def on_llm_end(self, response, **kwargs):
        try:
            gen = response.generations[0][0]
            if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                meta = gen.message.usage_metadata
                with self._lock:
                    self._tokens_in += meta.get("input_tokens", 0)
                    self._tokens_out += meta.get("output_tokens", 0)
        except (IndexError, TypeError, AttributeError):
            pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        with self._lock:
            self._tool_count += 1
            c = self._tool_count
        if c <= 5 or c % 10 == 0:
            name = serialized.get("name", "tool") if isinstance(serialized, dict) else "tool"
            self._log(f"🔧 {name} call #{c}...")

    def summary(self) -> str:
        with self._lock:
            return (
                f"LLM calls: {self._llm_count}, "
                f"Tool calls: {self._tool_count}, "
                f"Tokens: {self._tokens_in} in / {self._tokens_out} out"
            )


# ---------------------------------------------------------------------------
# Global analysis state
# ---------------------------------------------------------------------------
_analysis_result: Optional[Dict[str, Any]] = None
_analysis_error: Optional[str] = None
_analysis_running = False
_analysis_progress: list = []
_analysis_handler: Optional[WebProgressHandler] = None


def _run_analysis(ticker: str, date_str: str):
    """Run TradingAgents analysis in a background thread."""
    global _analysis_result, _analysis_error, _analysis_running, _analysis_progress, _analysis_handler

    _analysis_progress = []
    handler = WebProgressHandler(_analysis_progress)
    _analysis_handler = handler

    handler._log(f"🚀 Starting analysis for {ticker} on {date_str}...")

    try:
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG.copy()
        handler._log(f"⚙️ Provider: {config.get('llm_provider')}")
        handler._log(f"⚙️ Deep think: {config.get('deep_think_llm')}")
        handler._log(f"⚙️ Quick think: {config.get('quick_think_llm')}")
        handler._log(f"⚙️ Language: {config.get('output_language')}")

        # Create graph with our callback handler
        ta = TradingAgentsGraph(
            debug=False,
            config=config,
            callbacks=[handler],
        )

        handler._log("📊 Running multi-agent analysis pipeline...")
        handler._log("   Agents: Market → News → Sentiment → Fundamentals → Research → Trader → Risk → Portfolio")

        # Run the full pipeline
        state, decision = ta.propagate(ticker, date_str)

        handler._log(f"✅ Analysis complete! Decision: {decision}")
        handler._log(f"📊 {handler.summary()}")

        # Build result
        result = {
            "ticker": ticker,
            "date": date_str,
            "decision": decision,
            "market_report": state.get("market_report", ""),
            "sentiment_report": state.get("sentiment_report", ""),
            "news_report": state.get("news_report", ""),
            "fundamentals_report": state.get("fundamentals_report", ""),
            "investment_plan": state.get("investment_plan", ""),
            "trader_investment_plan": state.get("trader_investment_plan", ""),
            "final_trade_decision": state.get("final_trade_decision", ""),
            "bull_history": state.get("investment_debate_state", {}).get("bull_history", ""),
            "bear_history": state.get("investment_debate_state", {}).get("bear_history", ""),
            "aggressive_history": state.get("risk_debate_state", {}).get("aggressive_history", ""),
            "conservative_history": state.get("risk_debate_state", {}).get("conservative_history", ""),
            "neutral_history": state.get("risk_debate_state", {}).get("neutral_history", ""),
        }
        _analysis_result = result

        # Auto-save to history
        try:
            record_id = _save_to_history(ticker, date_str, decision, result)
            handler._log(f"💾 Saved to history (id: {record_id})")
        except Exception as e:
            handler._log(f"⚠️ Failed to save history: {e}")

    except Exception as e:
        _analysis_error = str(e)
        handler._log(f"❌ Error: {e}")
        import traceback
        for line in traceback.format_exc().splitlines():
            handler._log(f"  {line}")
    finally:
        _analysis_running = False


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    def mask_key(key: str) -> str:
        if not key or key in ("", "your-deepseek-api-key-here", "your-openai-api-key-here"):
            return ""
        if len(key) > 12:
            return key[:8] + "*" * (len(key) - 12) + key[-4:]
        return "***"

    return jsonify({
        "has_deepseek": bool(deepseek_key and deepseek_key not in ("", "your-deepseek-api-key-here")),
        "masked_deepseek": mask_key(deepseek_key),
        "has_openai": bool(openai_key and openai_key not in ("", "your-openai-api-key-here")),
        "masked_openai": mask_key(openai_key),
        "llm_provider": os.environ.get("TRADINGAGENTS_LLM_PROVIDER", "deepseek"),
        "deep_think": os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM", "deepseek-v4-pro"),
        "quick_think": os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM", "deepseek-v4-flash"),
        "language": os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE", "中文"),
    })


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    global _analysis_running, _analysis_result, _analysis_error, _analysis_progress

    if _analysis_running:
        return jsonify({"error": "Analysis already in progress"}), 429

    data = request.get_json() or {}
    ticker = data.get('ticker', '').strip().upper()
    date_str = data.get('date', '').strip()

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    _analysis_result = None
    _analysis_error = None
    _analysis_progress = []
    _analysis_running = True

    thread = threading.Thread(target=_run_analysis, args=(ticker, date_str), daemon=True)
    thread.start()

    return jsonify({"status": "started", "ticker": ticker, "date": date_str})


@app.route('/api/analysis-progress')
def api_analysis_progress():
    return jsonify({
        "running": _analysis_running,
        "progress": _analysis_progress[-50:] if _analysis_progress else [],
        "done": _analysis_result is not None,
        "error": _analysis_error,
    })


@app.route('/api/analysis-result')
def api_analysis_result():
    if _analysis_error:
        return jsonify({"error": _analysis_error}), 500
    if _analysis_result is None:
        return jsonify({"error": "No result yet"}), 404
    return jsonify(_analysis_result)


@app.route('/assets/<path:filename>')
def serve_assets(filename):
    assets_dir = os.path.join(_project_root, 'assets')
    return send_from_directory(assets_dir, filename)



@app.route('/api/history')
def api_history():
    """Get analysis history list."""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    records, total = _get_history(limit, offset)
    return jsonify({"records": records, "total": total})


@app.route('/api/history/<record_id>')
def api_history_detail(record_id):
    """Get full detail of a history record."""
    record = _get_history_detail(record_id)
    if record is None:
        return jsonify({"error": "Record not found"}), 404
    return jsonify(record)


@app.route('/api/history/<record_id>', methods=['DELETE'])
def api_history_delete(record_id):
    """Delete a history record."""
    _delete_history(record_id)
    return jsonify({"success": True})


@app.route('/api/history', methods=['DELETE'])
def api_history_clear():
    """Clear all history."""
    _clear_history()
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Workspace / Account Analysis API
# ---------------------------------------------------------------------------

_ACCOUNT_ANALYSIS_DIR = os.path.join(_project_root, 'web', 'artifacts', 'account_analysis')


def _build_workspace_response(workspace=None, workspace_name='default', created_at=None, updated_at=None):
    resp = normalize_workspace_payload(workspace or {})
    resp['workspace_name'] = workspace_name or 'default'
    resp['created_at'] = created_at
    resp['updated_at'] = updated_at
    return resp


def _account_analysis_output_dir(workspace_name):
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '-', str(workspace_name).strip() or 'default')
    return Path(_ACCOUNT_ANALYSIS_DIR) / safe_name


def _list_all_account_analysis_history():
    d = Path(_ACCOUNT_ANALYSIS_DIR)
    if not d.exists() or not d.is_dir():
        return []
    history = []
    for child in d.iterdir():
        if child.is_dir():
            history.extend(list_account_analysis_history(child))
    history.sort(key=lambda x: str(x.get('created_at') or ''), reverse=True)
    return history[:10]


def _utc_now_iso():
    from datetime import timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@app.route('/api/workspace', methods=['GET'])
def api_get_workspace():
    try:
        name = (request.args.get('workspace_name') or 'default').strip() or 'default'
        stored = db_manager.get_stage2_workspace(name)
        if stored:
            return jsonify({"success": True, "workspace": _build_workspace_response(
                workspace=stored,
                workspace_name=stored.get('workspace_name', name),
                created_at=stored.get('created_at'),
                updated_at=stored.get('updated_at'),
            )})
        return jsonify({"success": True, "workspace": _build_workspace_response(workspace_name=name)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/workspace', methods=['POST'])
def api_save_workspace():
    try:
        payload = request.get_json() or {}
        name = str(payload.get('workspace_name', 'default')).strip() or 'default'
        normalized = normalize_workspace_payload(payload)
        saved = db_manager.save_stage2_workspace(normalized, workspace_name=name)
        return jsonify({"success": True, "workspace": _build_workspace_response(
            workspace=saved,
            workspace_name=saved.get('workspace_name', name),
            created_at=saved.get('created_at'),
            updated_at=saved.get('updated_at'),
        )})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/account-analysis', methods=['POST'])
def api_run_account_analysis():
    try:
        payload = request.get_json() or {}
        name = str(payload.get('workspace_name', 'default')).strip() or 'default'

        stored = db_manager.get_stage2_workspace(name)
        if not stored:
            return jsonify({"success": False, "error": f"Workspace not found: {name}"}), 404

        # Refresh prices for all positions
        refreshed = recalculate_account_state(stored)
        saved = db_manager.save_stage2_workspace(refreshed, workspace_name=name)

        # Build analysis payload with basic metrics
        acct = saved.get("account_state", {})
        positions = saved.get("positions", [])
        analysis = {
            "summary": f"账户分析完成。共 {acct.get('position_count', 0)} 个持仓，"
                       f"NAV: {acct.get('nav', 0):.2f}, "
                       f"现金占比: {acct.get('cash_pct', 0):.1f}%, "
                       f"总敞口: {acct.get('gross_exposure', 0):.1f}%",
            "portfolio_health_score": _calc_health_score(acct, positions),
            "holding_health": [_calc_holding_health(p) for p in positions],
            "pnl_breakdown": {
                "total_unrealized_pnl": sum(p.get("unrealized_pnl", 0) for p in positions),
                "position_count": len(positions),
            },
            "concentration_risks": _calc_concentration_risks(acct, positions),
            "crowded_exposures": [],
            "manager_actions": _calc_manager_actions(acct, positions),
        }

        created_at = _utc_now_iso()
        markdown_body = _format_account_analysis_markdown(name, created_at, analysis)
        artifacts = save_account_analysis_artifacts(
            output_dir=_account_analysis_output_dir(name),
            workspace_name=name,
            analysis_payload=analysis,
            markdown_body=markdown_body,
            created_at=created_at,
        )

        return jsonify({
            "success": True,
            "workspace_name": name,
            "workspace": _build_workspace_response(
                workspace=saved,
                workspace_name=saved.get('workspace_name', name),
                created_at=saved.get('created_at'),
                updated_at=saved.get('updated_at'),
            ),
            "analysis": analysis,
            "analysis_markdown": markdown_body,
            "artifacts": artifacts,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/account-analysis/history', methods=['GET'])
def api_account_analysis_history():
    try:
        name = (request.args.get('workspace_name') or '').strip()
        if name:
            history = list_account_analysis_history(_account_analysis_output_dir(name))
        else:
            history = _list_all_account_analysis_history()
        return jsonify({"success": True, "history": history})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _calc_health_score(acct, positions):
    score = 50.0
    # Diversification
    n = len(positions)
    if n >= 5: score += 15
    elif n >= 3: score += 5
    else: score -= 10
    # Cash buffer
    cash_pct = acct.get("cash_pct", 0)
    if 10 <= cash_pct <= 40: score += 15
    elif cash_pct > 0: score += 5
    else: score -= 10
    # Drawdown
    dd = acct.get("current_drawdown", 0)
    if dd < 5: score += 10
    elif dd < 15: score += 5
    else: score -= 10
    # Largest position concentration
    lpw = acct.get("largest_position_weight", 0)
    if lpw > 40: score -= 15
    elif lpw > 25: score -= 5
    else: score += 10
    return max(0, min(100, round(score, 1)))


def _calc_holding_health(p):
    upnl_pct = p.get("unrealized_pnl_pct", 0)
    pw = p.get("position_weight", 0)
    issues = []
    if upnl_pct < -10:
        issues.append("深度亏损")
    elif upnl_pct < -5:
        issues.append("轻微亏损")
    if pw > 25:
        issues.append("权重过高")
    return {
        "ticker": p.get("ticker", ""),
        "unrealized_pnl_pct": round(upnl_pct, 2),
        "weight": round(pw, 2),
        "status": "健康" if not issues else "; ".join(issues),
    }


def _calc_concentration_risks(acct, positions):
    risks = []
    lpw = acct.get("largest_position_weight", 0)
    if lpw > 30:
        risks.append({"risk": f"最大持仓({acct.get('largest_position', '')})占比{lpw:.1f}%，集中度偏高"})
    if acct.get("cash_pct", 0) < 5:
        risks.append({"risk": "现金比例过低，缺乏缓冲"})
    if acct.get("gross_exposure", 0) > 95:
        risks.append({"risk": "总敞口过高"})
    return risks


def _calc_manager_actions(acct, positions):
    actions = []
    for p in positions:
        upnl = p.get("unrealized_pnl_pct", 0)
        pw = p.get("position_weight", 0)
        ticker = p.get("ticker", "")
        if upnl < -15 and pw > 20:
            actions.append({"action": f"考虑减仓 {ticker}", "reason": f"亏损{upnl:.1f}%且权重{pw:.1f}%"})
        elif upnl > 20 and pw < 5:
            actions.append({"action": f"考虑加仓 {ticker}", "reason": f"盈利{upnl:.1f}%但权重仅{pw:.1f}%"})
    if acct.get("cash_pct", 0) > 50:
        actions.append({"action": "考虑增加持仓", "reason": "现金比例过高"})
    return actions


def _format_account_analysis_markdown(workspace_name, created_at, analysis):
    lines = [
        "# 账户分析",
        "",
        f"- 工作区: {workspace_name}",
        f"- 生成时间: {created_at}",
        f"- 组合健康分: {analysis.get('portfolio_health_score', 0)}",
        "",
        "## 摘要",
        analysis.get("summary", ""),
    ]
    actions = analysis.get("manager_actions") or []
    lines.extend(["", "## 经理动作"])
    for a in actions:
        lines.append(f"- {a.get('action', '')}: {a.get('reason', '')}")
    if not actions:
        lines.append("- 无")

    health = analysis.get("holding_health") or []
    lines.extend(["", "## 持仓健康度"])
    for h in health:
        lines.append(f"- {h.get('ticker', '')}: {h.get('status', '')} (盈亏: {h.get('unrealized_pnl_pct', 0):.1f}%, 权重: {h.get('weight', 0):.1f}%)")

    risks = analysis.get("concentration_risks") or []
    lines.extend(["", "## 集中度风险"])
    for r in risks:
        lines.append(f"- {r.get('risk', '')}")
    if not risks:
        lines.append("- 无")

    pnl = analysis.get("pnl_breakdown") or {}
    lines.extend(["", "## 盈亏拆解"])
    for k, v in pnl.items():
        lines.append(f"- {k}: {v}")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TradingAgents Web Interface')
    parser.add_argument('--port', type=int, default=5001, help='Port (default: 5001)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host (default: 127.0.0.1)')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    print("=" * 60)
    print("  🤖 TradingAgents - Multi-Agent Trading Analysis Web")
    print("=" * 60)
    print(f"  URL:      http://{args.host}:{args.port}")
    print(f"  Provider: {os.environ.get('TRADINGAGENTS_LLM_PROVIDER', 'deepseek')}")
    print(f"  Deep:     {os.environ.get('TRADINGAGENTS_DEEP_THINK_LLM', 'deepseek-v4-pro')}")
    print(f"  Quick:    {os.environ.get('TRADINGAGENTS_QUICK_THINK_LLM', 'deepseek-v4-flash')}")
    print(f"  Language: {os.environ.get('TRADINGAGENTS_OUTPUT_LANGUAGE', '中文')}")
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print()

    app.run(debug=args.debug, host=args.host, port=args.port)
