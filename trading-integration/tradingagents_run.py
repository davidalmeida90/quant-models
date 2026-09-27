"""tradingagents run, from the write-up at
https://davidariasfinance.com/scripts/free-trading-tools/
"""

# git clone https://github.com/TauricResearch/TradingAgents.git && cd TradingAgents && pip install .
# provider comes from --provider or the TA_PROVIDER environment variable: openai, deepseek or ollama
# keys live in the environment, never in code: OPENAI_API_KEY or DEEPSEEK_API_KEY; Ollama needs none
#   ollama pull qwen3:8b            (for the local run)
import argparse
import os

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ap = argparse.ArgumentParser()
ap.add_argument("--provider", default=os.environ.get("TA_PROVIDER", "openai"),
                choices=["openai", "deepseek", "ollama"])
ap.add_argument("--ticker", default="NVDA")
ap.add_argument("--date", default="2026-09-01")
args = ap.parse_args()

config = DEFAULT_CONFIG.copy()
config["max_debate_rounds"] = 1              # 2 doubles the calls and rarely changes the call

if args.provider == "openai":                # paid API key, README defaults on 27 Sep 2026
    config["llm_provider"] = "openai"
    config["deep_think_llm"] = "gpt-6-sol"       # reasoning: research manager, portfolio manager
    config["quick_think_llm"] = "gpt-6-luna"     # fast: analysts, debates, tool calls
elif args.provider == "deepseek":            # paid API key, about 6 cents a full run, what the video uses
    config["llm_provider"] = "deepseek"
    config["deep_think_llm"] = "deepseek-v4-pro"     # the 2 managers
    config["quick_think_llm"] = "deepseek-v4-flash"  # everyone else
else:                                        # local, free, through Ollama's OpenAI compatible endpoint
    config["llm_provider"] = "ollama"
    config["backend_url"] = "http://localhost:11434/v1"
    config["deep_think_llm"] = "qwen3:8b"
    config["quick_think_llm"] = "qwen3:8b"

ta = TradingAgentsGraph(debug=True, config=config)
state, decision = ta.propagate(args.ticker, args.date)   # ticker, trade date
print(decision)                                            # BUY, SELL or HOLD with the reasoning
