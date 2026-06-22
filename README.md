# hippocampus-memory

本地优先的 AI 外部记忆与 Reasonix 上下文压缩系统。

这个项目不是普通 RAG。它的目标不是“把很多文本塞进向量库再搜索”，而是为 AI coding CLI 提供可审计、短小、按项目隔离的工作记忆：项目现状、历史决策、约束、失败经验、代码地图、影响范围和压缩后的 Context Bundle。

当前版本重点面向 Reasonix。其他 CLI 仍保留通用能力，但产品化优先级低于 Reasonix 集成。

## 当前状态

- 可用：本地 SQLite 记忆库、项目索引、Memory Pack、Project Profile、Code Map、Code Impact Pack、Context Bundle、自动存储、自动召回、轻量 MCP、Reasonix 一键部署。
- 可选：`jieba`/`rapidfuzz` 质量增强、`tiktoken` 计数、`sentence-transformers`、Chroma、`basedpyright` 诊断。
- 已知限制：Reasonix 状态栏里的 token 节省是“上下文压缩估算”，不是模型厂商账单实测。项目没有拦截 Reasonix 发给模型的真实请求，因此不能精确知道“如果不用 Hippo，这一轮真实会花多少 token”。

## 适合谁

- 经常用 Reasonix/Codex/Claude Code 做长周期项目的人。
- 项目上下文、决策、失败经验经常被 AI 忘掉的人。
- 想把“该读什么上下文”从手工复制粘贴变成自动调度的人。
- 需要本地优先、默认不上传记忆的人。

如果你的任务只是短问答、一次性脚本、普通文档检索，本项目会显得过重。

## 安装

要求：

- Windows PowerShell
- Python 3.11+
- 已安装 Reasonix，并且 `reasonix` 在 PATH 中

一键安装并部署当前项目：

```powershell
git clone https://github.com/1362909994-create/hippo_memory.git
cd hippo_memory
.\install-reasonix-hippo.ps1 -ProjectRoot D:\your_project -ProjectName your_project
reasonix code D:\your_project
```

安装脚本会做这些事：

- 安装 `hippocampus-memory`
- 给目标项目创建 `.hippo/hippo.db` 和 `.hippo.toml`
- 索引项目文件摘要、符号、imports 和调用线索
- 写入 Reasonix MCP 配置：`hippo_memory=hippo mcp-project`
- 安装全局 Reasonix shim，让 `reasonix` 启动时自动注入 Hippo Context Bundle
- 给 Reasonix 状态栏打补丁，显示会话级“预计节省”统计

如果机器没有 Python 3.11+，可以让脚本尝试通过 winget 安装：

```powershell
.\install-reasonix-hippo.ps1 -InstallPythonWithWinget
```

## 部署自检

部署后先跑一次只读诊断，确认项目库、Reasonix MCP 配置、全局 shim、状态栏补丁和全局提示块是否都在位：

```powershell
hippo doctor --root D:\your_project
hippo doctor --root D:\your_project --json
```

`ready: true` 表示 Reasonix 自动注入路径基本齐全。`recommendations` 会告诉你缺的是 `reasonix-deploy`、`reasonix-install-shim`、状态栏补丁、PATH，还是 Reasonix 本身。
如果 `--root` 指向项目子目录，doctor 会自动向上查找 `.hippo.toml` 或 `.hippo/hippo.db` 所在的项目根目录。

## 卸载

撤回这台机器上的 Reasonix/Hippo 集成：

```powershell
.\uninstall-reasonix-hippo.ps1 -ProjectRoot D:\your_project -RemoveProjectData -UninstallPackage
```

它会恢复 Reasonix 原始启动文件和 UI bundle，移除 `~\.reasonix\config.json` 里的 `hippo_memory` MCP 项，移除 `~\.reasonix\REASONIX.md` 里的 Hippo 提示块，并可选删除目标项目的 `.hippo/` 和 `.reasonix/` 本地数据。

默认不会修改项目里已跟踪的 `AGENTS.md`、`CLAUDE.md` 或 `REASONIX.md`。如果确实要移除项目提示块：

```powershell
.\uninstall-reasonix-hippo.ps1 -ProjectRoot D:\your_project -RemoveProjectMemory
```

## 常用命令

```powershell
hippo project-init my-project
hippo index-project D:\your_project --project my-project
hippo write --project my-project --type decision --content "Use SQLite as the default local store."
hippo search "previous decision about storage" --project my-project
hippo explain <memory_id> --project my-project --query "previous decision about storage"
hippo pack "continue the storage task" --project my-project
hippo project-profile --project my-project
hippo impact "change retrieval ranking" --project my-project
hippo auto-context "fix retrieval ranking bug" --project my-project --metadata
hippo auto-store --project my-project --text "Decision: rank exact project facts above generic source chunks."
hippo token-report "continue current task" --project my-project
hippo token-ledger --project my-project
hippo doctor --root D:\your_project --json
```

## Reasonix 工作方式

部署后，Reasonix 启动流程大致是：

1. 全局 `reasonix` shim 判断当前命令是否是 `reasonix code ...`。
2. shim 找到当前项目目录，必要时自动创建最小 `.hippo` 项目库。
3. `hippo reasonix-bootstrap-context` 根据当前项目生成短 Context Bundle。
4. shim 通过 `--system-append-file` 把 Context Bundle 注入 Reasonix。
5. Reasonix 里的 MCP 工具可按需调用 `hippo_memory_context_auto` 和 `hippo_memory_memory_auto_store`。
6. 状态栏读取 Hippo status JSON，按 Reasonix 会话单独显示预计节省。

状态栏统计口径：

- 新 Reasonix 会话从 0 开始。
- 打开旧 Reasonix 会话时，读取该会话自己的 ledger。
- 只有 Reasonix 进入真实对话轮次后，才把本次 Context Bundle 的估算节省计入会话。
- `预计节省` 是 `baseline_tokens - output_tokens`，其中 baseline 来自项目已索引记忆和文件摘要，output 是实际注入的 Context Bundle。
- 这不是 DeepSeek/Reasonix 的精确账单值。

## 核心产物

### Memory Pack

短上下文包，面向当前任务召回长期记忆。优先包含约束、决策、失败经验、任务状态和确认事实，默认排除 private/sensitive 记忆。

### Project Profile

项目级摘要，包括目标、当前状态、索引规模、功能概览、风险、未知点和最近记忆。

### Code Map

基于项目索引生成的代码地图，包含文件摘要、符号、imports 和相关 chunks。它不是完整 LSP，但足够给 AI 快速定位文件。

### Code Impact Pack

改代码前使用。根据当前 intent、项目索引、符号、调用线索和记忆，给出可能影响文件、风险、不变量、最小改动方向和建议测试。

### Context Bundle

组合 Project Profile、Memory Pack、Code Impact Pack 和 Code Map，是 Reasonix 自动注入的主要内容。

## 自动存储和自动召回

自动存储：

- 高置信、非敏感的长期事实直接写入。
- 中置信或敏感内容进入 candidate queue。
- 闲聊、重复日志、低价值临时内容跳过。

自动召回：

- 小闲聊不召回。
- 继续任务召回 compact callback pack。
- 调试和代码修改召回 lean Context Bundle。
- 项目综述召回 full Context Bundle。
- 显式记忆查询召回 Memory Pack。

## 数据和隐私

- 默认使用本地 SQLite。
- 默认不召回 private/sensitive 记忆。
- 项目记忆按 project 隔离，不会自动混入其他项目。
- 项目索引不复制完整源码，只存路径、hash、摘要、符号、imports、调用线索和 chunks。
- LLM summarizer、Chroma、sentence-transformers 都是可选能力，默认路径不要求联网。

## API 和 MCP

HTTP API：

```powershell
hippo serve --host 127.0.0.1 --port 8765
```

轻量 MCP：

```powershell
hippo mcp
hippo mcp-project
```

当前 MCP 是 JSON-RPC stdio 轻量实现，工具语义已稳定，但还不是完整 MCP SDK 产品。

## 测试

本仓库当前测试覆盖：

- 数据库 schema、CRUD、soft/hard delete
- 检索、重排、去重、敏感过滤
- Memory Pack、Context Bundle、Project Profile、Impact Pack
- 项目索引、Python AST 符号提取、调用线索
- 自动存储、自动召回策略
- Reasonix 部署、shim、状态栏补丁、卸载恢复
- CLI 行为、token ledger、安装脚本存在性

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check hippocampus_memory tests
```

## 路线图

优先级最高：

- 做真实 Reasonix 端到端 UI 测试。
- 把 token 节省统计和项目 token ledger 分层，避免估算值被误解成账单值。
- 建立更严谨的 memory admission gate，减少错误记忆进入上下文。
- 扩展 eval benchmark，覆盖中文项目、长会话、多轮调试、敏感泄漏和冲突记忆。

中期：

- 完整 MCP SDK 适配。
- 更强的代码图和影响分析。
- 对比 Mem0/Zep/Letta/PROJECTMEM 的公开 benchmark 设计自己的评测集。

低优先级或应砍掉：

- 非 Reasonix CLI 的过早产品化。
- 独立 browser/daemon 的重 UI 化。
- 过多注入模式导致的维护面扩大。

更完整的项目评估见 [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)。
