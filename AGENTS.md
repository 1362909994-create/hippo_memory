# AGENTS.md

本项目是 `hippocampus-memory`，目标是实现一个本地优先的 AI 外部记忆与 vibe coding 上下文压缩系统，用来给 Codex、Claude Code、DeepSeek、本地 Agent 等工具提供短小、准确、可审计的上下文。

维护规则：

- 不要把它做成普通 RAG；它的核心产物是 Memory Pack、Project Profile、Code Map、Code Graph、Code Impact Pack 和 Context Bundle。
- 召回质量比存储数量更重要，新增功能要优先考虑去重、重排、压缩、过期和冲突。
- Memory Pack 必须短、准、稳定，避免把完整历史聊天或大量原始代码塞进去。
- Project Profile 用来帮助 AI 快速理解整体项目、当前功能、风险点和未知点。
- Code Impact Pack 用来帮助 AI 判断影响范围、风险、不变量、最小改动方向和应跑测试。
- `hippo run` 是第一版自动喂入口：先生成 Context Bundle，再用 print/file/env/stdin/arg 注入给外部 AI coding 命令。
- `hippo run` 默认记录 session event；写入 session memory 必须显式 `--write-session-memory --yes`。
- `hippo project-init` 会写 `.hippo.toml`，后续命令应优先用项目配置自动识别 project。
- 会话摘要应优先进入 candidate queue，再由用户 `candidate-accept` 写入长期记忆。
- `hippo mcp` 是第一版 callback 入口，但当前是轻量 JSON-RPC stdio 服务；升级完整 MCP 时要保留现有工具语义。
- `hippo eval` 用 JSONL benchmark 检查召回质量；涉及检索/Pack 的改动应补评估样例。
- eval 用例支持 `mode=pack`、`forbidden_contains` 和 `max_tokens`；涉及上下文包裁剪/敏感过滤时必须覆盖。
- `hippo token-report` 用来估算 Context Bundle 相对朴素上下文的 token 节省，相关变更要保持估算保守。
- Chroma、sentence-transformers、LLM summarizer 都必须是可选能力；默认路径不能要求联网或重依赖。
- LLM 摘要失败必须回退规则摘要，且写入长期记忆仍需确认。
- `hippo mcp-config` 和 `hippo daemon-script` 是部署辅助入口；不要把它们做成破坏性安装器。
- `hippo browser` 只能展示非敏感记忆，不要默认暴露 private/sensitive 内容。
- 冲突不要自动覆盖事实；用 `conflict-list` / `conflict-resolve` 进入显式解决流程。
- 不要默认召回 `sensitive` 或 `private` 记忆，除非调用方显式请求。
- 会话摘要默认只能预览；写入长期记忆必须显式确认，避免把猜测或临时讨论写成事实。
- 做代码影响分析时优先复用 `files`、`chunks`、`memories`、`conflicts` 里的结构化信息，不要无脑读取或复制整个项目。
- Python 项目索引优先使用标准库 AST 提取 symbols/imports/calls；其他语言仍是正则 fallback，不要把影响分析结果当成绝对调用图。
- 代码修改前先运行相关测试；新功能必须加测试。
- 保持 Windows 路径兼容，路径处理统一使用 `pathlib`。
- 不要引入过重依赖。FAISS、Chroma、sentence-transformers 等应保持可选。
- 不要破坏现有 API、CLI 命令和数据库迁移兼容性。
- 猜测不能写成事实，低置信度信息的 `confidence` 不应超过 `0.6`。
- 删除逻辑必须区分 soft delete 和 hard delete；用户明确要求删除时要支持彻底删除。

<!-- hippocampus-memory:start -->
## Hippocampus Memory

- This project is deployed with project-local hippocampus-memory as `code-hippo-memory`.
- The MCP server name is `hippo_memory`; prefer its automatic tools when external project memory, code impact analysis, or compact context would help.
- At the start of non-trivial coding/debugging/architecture tasks, call `hippo_memory_context_auto` with the current task intent. Use `session_key="codex"` unless the user gives a better session name. Trust the tool when it returns that no external memory is needed.
- For direct symbol questions use `hippo_memory_code_symbols` or `hippo_memory_code_references`; otherwise prefer `hippo_memory_context_auto` over manually choosing profile, impact, callback, or bundle tools.
- Near the end of meaningful work, call `hippo_memory_memory_auto_store` with a concise transcript summary. It will write high-confidence non-sensitive memories, queue uncertain memories, and skip low-value content.
- Do not recall private/sensitive memories unless explicitly requested. Do not force long-term writes for sensitive or uncertain facts.
- Keep recalled context short, cite files when making code claims, make minimal changes, and run relevant tests.
<!-- hippocampus-memory:end -->
