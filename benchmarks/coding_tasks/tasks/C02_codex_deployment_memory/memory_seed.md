# Memory seed for C02

Prior issue: real `codex mcp list` failed because the user-level `C:\Users\dell\.codex\config.toml` contained `service_tier = "default"`, while `codex-cli 0.130.0` accepted `fast` or `flex`. The benchmark should not mutate global Codex config automatically.

Deployment rule: `hippo codex-deploy` writes project-local `.hippo/hippo.db`, `.hippo.toml`, `.hippo/codex-mcp-config.json`, and an AGENTS.md Hippo block. `hippo doctor --json` is read-only and should report problems clearly without editing files.

Regression warning: do not add back `reasonix-deploy`, Reasonix token UI hooks, or global background token monitors.
