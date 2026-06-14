#!/usr/bin/env python3
"""TradingAgents Web Interface - Quick Launcher"""

import os
import sys
from pathlib import Path

# Ensure we're in the project root
os.chdir(Path(__file__).parent)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from web.app import app

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
