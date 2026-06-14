"""Account analysis history storage - port of ElonQuantAgent's services/account_analysis.py"""

from __future__ import annotations

import json
from pathlib import Path


MAX_ACCOUNT_ANALYSIS_RUNS = 10


def save_account_analysis_artifacts(
    output_dir: Path,
    workspace_name: str,
    analysis_payload: dict,
    markdown_body: str,
    created_at: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_created_at = str(created_at or "").replace(":", "-")
    json_path = output_dir / f"{safe_created_at}.json"
    markdown_path = output_dir / f"{safe_created_at}.md"

    payload = dict(analysis_payload or {})
    payload["created_at"] = str(created_at or "")
    payload["workspace_name"] = str(workspace_name or "")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with markdown_path.open("w", encoding="utf-8") as f:
        f.write(str(markdown_body or ""))

    _prune_account_analysis_history(output_dir)

    return {
        "created_at": payload["created_at"],
        "workspace_name": payload["workspace_name"],
        "summary": _coerce_summary(payload),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }


def list_account_analysis_history(output_dir: Path) -> list[dict]:
    entries = _collect_account_analysis_history(output_dir)
    return entries[:MAX_ACCOUNT_ANALYSIS_RUNS]


def _prune_account_analysis_history(output_dir: Path) -> None:
    all_entries = _collect_account_analysis_history(output_dir)
    for entry in all_entries[MAX_ACCOUNT_ANALYSIS_RUNS:]:
        for key in ("json_path", "markdown_path"):
            p = entry.get(key)
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    continue


def _collect_account_analysis_history(output_dir: Path) -> list[dict]:
    if not output_dir.exists() or not output_dir.is_dir():
        return []
    entries = []
    for json_path in output_dir.glob("*.json"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            payload = {}
        created_at = str(payload.get("created_at") or json_path.stem)
        workspace_name = str(payload.get("workspace_name") or "")
        markdown_path = output_dir / f"{json_path.stem}.md"
        entries.append({
            "created_at": created_at,
            "workspace_name": workspace_name,
            "summary": _coerce_summary(payload),
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
        })
    entries.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    return entries


def _coerce_summary(payload: dict) -> str:
    raw = payload.get("summary") or payload.get("analysis", {}).get("summary") or ""
    text = str(raw).strip()
    if len(text) > 200:
        text = text[:200] + "..."
    return text
