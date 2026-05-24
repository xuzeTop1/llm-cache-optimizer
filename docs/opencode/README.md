# OpenCode Plugin — llm-cache-optimizer MCP Server

Expose prompt optimization, canonicalization, and cache metrics to OpenCode via MCP.

## Install

```bash
# 1. Install the Python package
cd llm-cache-optimizer
pip install -e .

# 2. Copy config to your project root (or global ~/.opencode/)
cp docs/opencode/opencode.json.example ./opencode.json
```

## How It Works

OpenCode supports MCP (Model Context Protocol) servers. This plugin starts a local MCP server that exposes 4 tools:

| Tool | Description |
|------|-------------|
| `optimize_prompt` | Reorder messages for prefix caching |
| `canonicalize_output` | Normalize tool output JSON (sorted keys, stripped timestamps) |
| `refresh_memory` | Compress conversation history into session memory |
| `cache_report` | Analyze prompt for cache hit rate and cost savings |

OpenCode automatically discovers these tools and makes them available to the LLM.

## Verify

```bash
# Test MCP server manually
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python -m llm_cache_optimizer.mcp_server
```

## Architecture

```
┌─────────────┐     JSON-RPC      ┌─────────────────────┐
│   OpenCode  │ ◄──── stdio ────► │  MCP Server (Python) │
│   (client)  │                    │  llm-cache-optimizer │
└─────────────┘                    └─────────────────────┘
```

Zero-config: the server runs as a subprocess, communicates over stdin/stdout, and requires no network access.
