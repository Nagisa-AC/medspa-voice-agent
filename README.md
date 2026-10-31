# medspa-voice-agent

| Target | What it does |
| --- | --- |
| `make sync` | Install exactly what `uv.lock` pins |
| `make lint` | Ruff lint and format check, no changes |
| `make fmt` | Auto-fix lint and formatting |
| `make typecheck` | mypy strict |
| `make test` | pytest |
| `make eval` | Eval harness (ScriptedLLM cases; fails the build on any failure) |
| `make check` | Everything CI runs |
