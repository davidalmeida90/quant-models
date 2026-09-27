# TradingView MCP server, install and register

Server by atilaahmettaner, MIT, no TradingView account or key needed:
[github.com/atilaahmettaner/tradingview-mcp](https://github.com/atilaahmettaner/tradingview-mcp).
Lines below are copied from its README on 27 September 2026.

## Install, pick one

```bash
pip install tradingview-mcp-server
uv tool install tradingview-mcp-server
```

## Claude Code, one line

```bash
claude mcp add tradingview -- uvx --from tradingview-mcp-server tradingview-mcp
```

Then `/mcp` inside a session to confirm it is connected.

## Claude Desktop and Cursor

Paste [tradingview_mcp.json](tradingview_mcp.json) into `claude_desktop_config.json`. Path to uvx must be
absolute there, because the app does not inherit your shell PATH. On Windows it is
`%USERPROFILE%\.local\bin\uvx.exe`.

## Optional, for the news and sentiment tools

Free tier, 100 requests a day:

```bash
export MARKETAUX_API_TOKEN=your_token_here
```
