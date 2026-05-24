## Highlights

### New: CacheAwareClaude Adapter
- Explicit `cache_control` breakpoints on stable layers for Anthropic Claude
- Automatically merges small layers to meet Claude's 1024-token minimum
- Install with: `pip install -e ".[anthropic]"`

### New: GitHub Actions CI
- Pytest runs on Python 3.9 / 3.10 / 3.11 / 3.12
- CI badge in README

### New: Benchmark Script
- DeepSeek-based benchmark comparing naive vs optimized cache hit rates
- Outputs CSV + matplotlib chart

### Other Changes
- `SessionMemory` now supports custom LLM summarizer callable
- `CacheMetrics.from_provider()` with pricing presets for DeepSeek, OpenAI, Claude
- Install instructions fixed: `git clone` + `pip install -e .` (source install)
- SKILL.md moved to `docs/skills/` to keep root clean
- Examples modernized to use library API

### Install
```bash
git clone https://github.com/xuzeTop1/llm-cache-optimizer.git
cd llm-cache-optimizer
pip install -e ".[all]"
```

**Full Changelog**: https://github.com/xuzeTop1/llm-cache-optimizer/compare/v0.2.0...v0.3.0
