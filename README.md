# hippocampus-memory

Reasonix-first external memory and token-saving context for AI coding.

`hippocampus-memory` 是一个本地优先的 AI 外部记忆与 vibe coding 上下文压缩系统。它不是普通 RAG 问答工具，而是给 Codex、Claude Code、DeepSeek、本地 Agent 等工具使用的“外部海马体”：把长期偏好、项目状态、历史决策、失败经验、约束、代码语义和影响范围保存到本地，再按当前任务召回、重排、压缩成短小的上下文包。

核心目标不是“存很多东西”，而是精准召回、去重、压缩、标记过期、处理冲突，并帮助 AI 用最少 token 理解项目、选择最小改动方向。

## 和普通 RAG 的区别

- 普通 RAG 通常围绕文档问答；本项目围绕 Agent 工作记忆、长期记忆和代码变更上下文。
- 普通 RAG 倾向返回资料片段；本项目生成给 AI 阅读的压缩上下文包。
- 记忆有类型、状态、可见性、置信度、重要性、项目归属和过期策略。
- 默认不召回 `sensitive` / `private` 记忆，避免把敏感内容塞进上下文。
- 代码索引不复制原始项目，只保存路径、hash、摘要、符号、import 和 chunk。

## 安装

如果你只想在 Reasonix 里自动使用外部记忆，推荐走一键入口：

```powershell
git clone https://github.com/1362909994-create/hippo_memory.git
cd hippo_memory
.\install-reasonix-hippo.ps1 -ProjectRoot D:\your_project -ProjectName your_project
reasonix code D:\your_project
```

完成一次全局 shim 安装后，之后也可以直接在其他项目目录运行 `reasonix` 或
`reasonix code D:\another_project`。Reasonix 启动时会自动生成本轮 context/status；
安全且可写的新项目目录会首次自动创建 `.hippo\hippo.db` 和 `.hippo.toml`，
底部状态栏按 Reasonix 会话单独显示 `本轮` 和 `会话` token 节省。

脚本会安装 `hippo`、部署项目记忆、配置 Reasonix MCP、安装全局 shim，并给 Reasonix 底部状态栏加上按会话统计的 token 节省显示。

开发本项目时再使用 editable 安装：

```powershell
cd D:\prj\hippocampus-memory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

### Reasonix 一键安装入口

用户从 GitHub 下载本仓库后，可以在仓库根目录直接运行：

```powershell
.\install-reasonix-hippo.ps1 -ProjectRoot D:\your_project -ProjectName your_project
```

这个脚本会安装 `hippocampus-memory`、安装 Reasonix 全局 shim 和状态栏补丁，并对目标项目执行 `hippo reasonix-deploy`。
如果只是想部署当前目录，可以省略参数：

```powershell
.\install-reasonix-hippo.ps1
```

如果机器没有 Python 3.11+，可以让脚本尝试用 winget 安装：

```powershell
.\install-reasonix-hippo.ps1 -InstallPythonWithWinget
```

全局 shim 会同时覆盖 npm 生成的 `reasonix.ps1`、`reasonix.cmd` 和无扩展名
`reasonix` 三个启动入口。对 `C:\Windows`、磁盘根目录、用户 Home 这类不适合
自动写项目文件的位置，不会创建 `.hippo`，但状态栏仍会从 0 显示，避免误以为
集成没有生效。

第一版为了 Windows 上稳定运行，没有强制安装 FAISS/Chroma 或 sentence-transformers。代码保留了 `EmbeddingBackend` 和 `VectorStore` 抽象，默认使用本地 hash embedding + SQLite JSON 向量降级实现；以后可以替换成 FAISS、Chroma、sentence-transformers、OpenAI 或本地模型。

可选启用 sentence-transformers：

```powershell
pip install -e .[semantic]
$env:HIPPO_EMBEDDING_BACKEND = "sentence-transformers"
$env:HIPPO_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

如果模型不可用，系统会自动降级到本地 hash embedding，keyword search 仍可用。

可选启用 Chroma 向量库：

```powershell
pip install -e .[chroma]
$env:HIPPO_VECTOR_BACKEND = "chroma"
$env:HIPPO_CHROMA_PATH = "D:\data\hippo-chroma"
```

如果 Chroma 不可用，系统会回退到 SQLite JSON 向量存储。

可选启用检索质量增强：

```powershell
pip install -e .[quality]
```

`quality` 会安装 `jieba` 和 `rapidfuzz`。启用后，中文查询会优先使用词级分词并保留 n-gram 回退，Memory Pack 的近重复去重也会更稳；如果这些库不可用，系统会自动回退到内置轻量逻辑。

可选启用 LSP / tokenizer 增强：

```powershell
pip install -e .[lsp,tokens]
```

`lsp` 会安装 `basedpyright`，用于 `hippo code-diagnostics --refresh` 采集 Python 类型、导入和未定义变量诊断；没有安装时系统继续使用内置 AST 准 LSP。`tokens` 会安装 `tiktoken`，让 `token-report --model ...` 更接近目标模型 tokenizer。

## 快速开始

```powershell
hippo init
hippo write --project glasses --type constraint --content "用户不接受 3-4 cm 焦距的光学结构。"
hippo write --project glasses --type task_state --content "当前目标是先让 STM32 点亮 TFT 屏幕。"
hippo search "继续上次那个屏幕项目" --project glasses
hippo pack "继续上次那个 STM32 点亮 TFT 的项目" --project glasses
```

默认数据库保存在当前用户目录的 `.hippocampus-memory\hippocampus.db`。也可以用 `HIPPO_DB_PATH` 指定：

```powershell
$env:HIPPO_DB_PATH = "D:\data\hippo.db"
```

## CLI

```powershell
hippo init
hippo project-init hippocampus-memory
hippo serve --host 127.0.0.1 --port 8765
hippo write --project "glasses-display" --type constraint --content "用户不接受 3-4 cm 焦距的直线光学结构。"
hippo search "TFT 屏幕不亮可能是什么原因" --project "glasses-display"
hippo pack "继续上次那个 STM32 点亮 TFT 的项目" --project "glasses-display"
hippo callback "继续上次那个 STM32 点亮 TFT 的项目" --project "glasses-display" --session codex
hippo callback-reset --project "glasses-display" --session codex
hippo index-project "D:/codex/prj1" --project "prj1"
hippo project-profile --project "prj1"
hippo code-map --project "prj1" --query "retriever search"
hippo code-symbols --project "prj1" --query "search"
hippo code-references search --project "prj1"
hippo code-intelligence "修改 search 排序" --project "prj1"
hippo code-diagnostics --project "prj1" --refresh
hippo impact "给搜索加入 entity boost" --project "prj1"
hippo run --project "prj1" --intent "给搜索加入 entity boost"
hippo code-graph --project "prj1"
hippo eval .\examples\retrieval_eval.jsonl
hippo browser --project "prj1" --output .\hippo-memory-browser.html
hippo token-report "给搜索加入 entity boost" --project "prj1"
hippo token-ledger --project "prj1"
hippo mcp-config --output .\hippo-mcp-config.json
hippo daemon-script --output .\start-hippo-daemon.ps1
hippo consolidate --project "glasses-display"
hippo memory-supersede mem_old mem_new
hippo stats
```

## 自动记忆与自动调度

部署后的推荐入口是自动策略，而不是让宿主 AI 手动选择每一个底层命令。

自动存储会从会话摘要或任务结果里筛选长期有价值的内容：高置信、非敏感的偏好、约束、决策、失败经验、任务状态会直接写入；中等置信或敏感内容默认进入候选队列；低价值闲聊、重复日志和临时噪音会跳过。

```powershell
hippo auto-store --project hippocampus-memory --text "Decision: use context.auto as the default recall entry."
hippo auto-store --project hippocampus-memory --path .\chat-summary.txt --mode preview
hippo candidate-list --project hippocampus-memory
hippo candidate-accept <candidate-id>
```

自动召回会根据当前 intent 决定是否需要外部记忆，以及返回哪种上下文：小闲聊不召回；继续任务返回 compact callback pack；代码修改/调试返回 lean context bundle；项目概览返回 full context bundle；显式记忆查询返回 Memory Pack。

```powershell
hippo auto-context "continue" --project hippocampus-memory
hippo auto-context "fix search ranking bug" --project hippocampus-memory --metadata
```

`pack`、`auto-context` 和 `run` 默认会在 CLI 中显示 token 账本：本次输出相对朴素上下文估算节省了多少 token，以及该项目历史累计节省了多少 token。统计会写入项目级 `token_ledger`，上下文本体仍正常输出；不想记录时可加 `--no-token-stats`。

```powershell
hippo pack "continue" --project hippocampus-memory
hippo auto-context "fix search ranking bug" --project hippocampus-memory --token-model gpt-4o
hippo token-ledger --project hippocampus-memory
```

MCP 部署时优先使用高层工具：

- `memory.auto_store` / safe name `memory_auto_store`
- `context.auto` / safe name `context_auto`

`reasonix-deploy` 写入的项目提示也会引导宿主 AI 优先调用 `context_auto`，并在有意义的会话结束时调用 `memory_auto_store`。

常用降 token / 去重参数：

```powershell
hippo search "TFT 屏幕不亮可能是什么原因" --project "glasses-display" --no-dedupe
hippo pack "继续上次那个 STM32 点亮 TFT 的项目" --project "glasses-display" --compact
hippo pack "继续上次那个 STM32 点亮 TFT 的项目" --project "glasses-display" --source-chunk-limit 0
hippo pack "继续上次那个 STM32 点亮 TFT 的项目" --project "glasses-display" --exclude-memory-id mem_xxx
```

默认 `search` 会合并近重复结果，避免同一事实占满 top-k；`--no-dedupe` 用于调试原始召回。`--compact` 适合小任务，会减少候选数、source chunk 和解释性模板。`--exclude-memory-id` 可由 CLI / GUI / MCP 调用方传入已经注入过的记忆 ID，避免长会话里反复把同一条记忆喂给 AI。

默认按项目隔离记忆：传入 `--project` 后，搜索、Pack 和报告只使用该项目的记忆，不会自动混入其他项目或空项目的“全局”记忆。

也支持：

```powershell
python -m hippocampus_memory search "项目状态" --project glasses
```

## 自动项目识别

在项目根目录写入 `.hippo.toml`：

```powershell
hippo project-init hippocampus-memory
```

之后在该目录或子目录运行这些命令时，可以省略 `--project`：

```powershell
hippo project-profile
hippo impact "修改搜索评分逻辑"
hippo run --intent "继续当前任务"
```

如果没有 `.hippo.toml`，系统会尝试用 Git root 目录名，否则使用当前目录名。

## 上下文包

### Memory Pack

```powershell
hippo pack "继续上次那个 STM32 点亮 TFT 的项目" --project glasses
```

Memory Pack 是给 AI 阅读的短上下文，不是搜索结果列表。它会优先包含项目状态、约束、失败记录、确认事实、开放问题和下一步建议。

如果只是一个很小的改动，可以用 `--compact` 生成更短的包；如果当前任务不需要源代码片段，可以把 `--source-chunk-limit` 设为 `0`。长会话或 GUI 集成中，调用方可以记录上一次已经注入过的 memory id，并在下一次 `pack` 时通过 `--exclude-memory-id` 或 MCP 的 `exclude_memory_ids` 传回，从外部控制“不要重复喂同一段记忆”。

也可以直接使用项目级 callback 会话：

```powershell
hippo callback "继续当前任务" --project hippocampus-memory --session codex
hippo callback "继续当前任务" --project hippocampus-memory --session codex --metadata
hippo callback-reset --project hippocampus-memory --session codex
```

`callback` 会记住该 project/session 已经注入过的 memory id，下次自动排除，避免同一段记忆反复喂给 AI。这个状态按项目和 session 隔离，不是全局记忆。

### Project Profile

```powershell
hippo project-profile --project hippocampus-memory
```

Project Profile 用来压缩整体项目理解：

- 项目目标 / 背景
- 当前实现形态
- 已索引文件和语言分布
- 功能清单
- 当前状态、决策、约束、失败记录
- 风险点和未知点

### Code Map

```powershell
hippo code-map --project hippocampus-memory --query "memory pack"
```

Code Map 基于项目索引输出文件摘要、符号和 import，帮助 AI 快速知道“代码长什么样、哪些文件相关”。

### Code Impact Pack

```powershell
hippo impact "修改 Memory Pack 生成规则" --project hippocampus-memory
```

Code Impact Pack 面向 vibe coding 改动前的最小上下文：

- 当前改动意图
- 相关长期记忆
- 可能影响的文件
- 风险点 / 不变量
- 建议最小改动方向
- 建议测试

### Context Bundle / 自动喂

```powershell
hippo run --project hippocampus-memory --intent "修改搜索评分逻辑"
```

默认 `--inject print`，只打印一个完整的 Context Bundle。它会组合：

- Project Profile
- Memory Pack
- Code Impact Pack
- Code Map

如果要启动另一个 AI coding 工具，可以把命令放在 `--` 后面：

```powershell
hippo run --project hippocampus-memory --intent "修改搜索评分逻辑" --inject stdin -- codex
```

注入模式：

- `print`：只打印上下文，不启动命令。
- `file`：把上下文写入临时文件，并设置 `HIPPO_CONTEXT_FILE` 后启动命令。
- `env`：设置 `HIPPO_CONTEXT`、`HIPPO_CONTEXT_FILE`、`HIPPO_PROJECT`、`HIPPO_INTENT` 后启动命令。
- `stdin`：把上下文写到子进程 stdin。
- `arg`：把上下文作为最后一个命令参数追加给子进程。

不同 AI CLI 接受上下文的方式不一样，所以第一版提供多种注入模式，而不是绑死某一个工具。

`hippo run` 默认会记录一次 `session.run` event，包含 intent、命令、return code、输出摘录和 Git 状态。要把运行结果同时沉淀成长期记忆，必须显式确认：

```powershell
hippo run --project foo --intent "修复搜索排序" --write-session-memory --yes --inject stdin -- codex
```

### MCP / Callback 入口

```powershell
hippo mcp
hippo mcp-project
hippo mcp-config --output .\hippo-mcp-config.json
```

这是第一版轻量 JSON-RPC stdio 工具服务，提供 `memory.write`、`memory.search`、`memory.pack`、`project.profile`、`project.impact`、`context.bundle`。后续可以升级为完整 MCP 协议适配。

Reasonix 可以用一条命令部署项目本地记忆：

```powershell
cd D:\prj\vivado_mcp
hippo reasonix-deploy --project vivado_mcp
reasonix code .
```

`reasonix-deploy` 会创建 `.hippo\hippo.db`、写 `.hippo.toml`、索引项目、生成 `.hippo\hippo-mcp.ps1`，并默认把 `hippo_memory=hippo mcp-project` 追加到 `~\.reasonix\config.json`。这个配置只是 MCP 启动入口；实际记忆仍来自当前 Reasonix workspace 下的 `.hippo\hippo.db`，不会混成全局记忆。进入 Reasonix 后用 `/mcp` 可以看到 `hippo_memory_` 前缀的工具。

它还会写一个短的 Reasonix 项目记忆片段：优先追加到已有 `REASONIX.md` / `AGENTS.md` / `CLAUDE.md`，没有这些文件时创建 `REASONIX.md`。这个片段只告诉 Reasonix 何时主动调用 `hippo_memory_context_callback`、`hippo_memory_context_bundle` 和 `hippo_memory_project_impact`，避免每次都手动提醒。若不想写项目提示，可加 `--no-project-memory`。

如果不想改 Reasonix 全局配置：

```powershell
hippo reasonix-deploy --project vivado_mcp --no-install-global
reasonix code . --mcp (Get-Content .\.hippo\reasonix-mcp-spec.txt)
```

Reasonix 的会话续接和 hippo 记忆是两层东西：`reasonix code . --resume` 续接 Reasonix 的聊天历史；`.hippo\hippo.db` 是项目本地外部记忆，即使开新会话也还在。

### LLM Session Summarizer

默认会话摘要是规则版。可以配置 OpenAI-compatible endpoint 后显式启用 LLM 摘要：

```powershell
$env:HIPPO_LLM_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
$env:HIPPO_LLM_MODEL = "local-model"
$env:HIPPO_LLM_API_KEY = "optional"
hippo queue-session chat.txt --project hippocampus-memory --llm
```

LLM 调用失败时会回退到规则摘要。写入长期记忆仍然需要候选确认或显式 `--write --yes`。

### Code Graph

```powershell
hippo code-graph --project hippocampus-memory
```

基于索引里的 `symbols` 和 `calls` 推断轻量跨文件调用边，用来辅助影响分析。它不是完整 AST/LSP 调用图。

Python 文件会优先用标准库 `ast` 提取 class/function/import/call；其他语言继续使用正则 fallback。

### Code Intelligence

```powershell
hippo code-symbols --project hippocampus-memory --query callback_pack
hippo code-references callback_pack --project hippocampus-memory
hippo code-intelligence "修改 callback 去重" --project hippocampus-memory
hippo code-diagnostics --project hippocampus-memory --refresh
```

这是轻量的 Python 准 LSP 索引：记录函数/类定义位置、qualified name、签名、docstring、引用和调用位置。`hippo impact` 也会使用这些信息输出函数级影响范围。它目前不是完整 language server，不提供类型推断和重命名编辑；后续可接 pyright / tsserver。
安装 `basedpyright` 后，`code-diagnostics --refresh` 会运行 `basedpyright --outputjson` 并把 diagnostics 写入项目缓存；`hippo impact` 会优先显示相关文件的存储诊断。未安装时该命令返回 unavailable，不影响其他功能。

### Evaluation

```powershell
hippo eval .\bench.jsonl
```

JSONL 每行一个用例：

```json
{"query":"display task","project":"glasses","expected_contains":["TFT"]}
```

也可以检查 Pack：

```json
{"mode":"pack","query":"display task","project":"glasses","expected_contains":["TFT"],"forbidden_contains":["password"],"max_tokens":500}
```

用于检查召回是否命中预期记忆，后续可扩展成 token 节省和 Pack 质量评估。

### Token Savings Report

```powershell
hippo token-report "修改搜索评分逻辑" --project hippocampus-memory
hippo token-report "修改搜索评分逻辑" --project hippocampus-memory --model gpt-4o
hippo token-ledger --project hippocampus-memory
```

这个报告用估算 token 对比 Context Bundle 和“直接塞入索引摘要/记忆”的朴素成本，帮助判断是否真的在省 token。
`token-report` 默认会把本次估算写入项目级 token ledger；`token-ledger` 用来查看该项目的历史平均节省率、累计节省 token 和最近记录。安装 `tiktoken` 后，`--model` 会优先使用模型 tokenizer；否则自动回退到内置估算器。它是趋势账本，不等同于模型厂商的精确计费账单。

### Daemon Script

```powershell
hippo daemon-script --output .\start-hippo-daemon.ps1
```

生成一个启动本地 daemon 的 PowerShell 脚本。它不是正式 Windows Service installer，但可以作为开机启动/任务计划程序的入口。

### Memory Browser

```powershell
hippo browser --project hippocampus-memory --output .\hippo-memory-browser.html
```

生成本地 HTML 报告，展示 stats、projects 和最近的非敏感记忆。它不会默认展示 `sensitive` / `private` 内容。

## API

启动服务：

```powershell
hippo serve --host 127.0.0.1 --port 8765
```

已有接口：

- `GET /health`
- `POST /memory/write`
- `POST /memory/search`
- `POST /memory/pack`
- `POST /memory/consolidate`
- `POST /memory/forget`
- `POST /project/index`
- `GET /project/{project}/summary`
- `GET /project/{project}/profile`
- `POST /project/code-map`
- `POST /project/impact`
- `GET /project/list`
- `GET /candidate/list`
- `POST /candidate/accept`
- `POST /candidate/discard`
- `GET /conflict/list`
- `POST /conflict/resolve`
- `POST /session/summarize`
- `POST /session/queue`
- `GET /stats`

## 索引项目

```powershell
hippo index-project "D:/path/to/project" --project "my-project"
```

索引器默认跳过 `.git`、`node_modules`、`.venv`、`build`、`dist`、`target`、缓存目录、二进制文件和超过 1MB 的文件。它不会复制原始工程文件，只保存路径、hash、摘要、符号、import 和 chunk。

重新索引时，已经从项目里消失的文件会标记为 `missing`，不会继续作为 active 文件参与 Code Map / Impact Pack。索引器还会提取简单的 `symbols`、`imports` 和 `calls`，但第一版仍不是完整 AST。

## 写入记忆

```powershell
hippo write --project glasses --type decision --content "选择先验证 TFT 背光和供电，再排查 SPI 初始化。"
```

支持类型：

- `user_preference`
- `project_context`
- `decision`
- `failure`
- `constraint`
- `technical_fact`
- `task_state`
- `source_chunk`

## 删除记忆

软删除：

```powershell
hippo forget <memory-id>
```

彻底删除：

```powershell
hippo forget <memory-id> --hard
```

删除整个项目的记忆：

```powershell
hippo forget --project glasses --hard
```

## 接入 Codex / Claude Code / DeepSeek

推荐工作流：

1. 会话结束时用 `hippo summarize-session chat.txt --project <project>` 生成候选记忆。
2. 确认后写入长期记忆。
3. 新会话开始前用 `hippo project-profile` 和 `hippo pack` 生成整体和任务上下文。
4. 改代码前用 `hippo impact "<改动意图>" --project <project>` 生成最小改动导航。
5. 把这些上下文包放到 Agent 的上下文开头。

为了避免污染长期记忆，`summarize-session` 默认只预览候选。如果要写入，必须显式确认：

```powershell
hippo summarize-session chat.txt --project hippocampus-memory --write --yes
```

更推荐的安全流程是先入候选队列，再人工确认：

```powershell
hippo queue-session chat.txt --project hippocampus-memory
hippo candidate-list --project hippocampus-memory
hippo candidate-accept <candidate-id>
hippo candidate-discard <candidate-id>
```

冲突也可以审查和解决：

```powershell
hippo conflict-list --project hippocampus-memory
hippo conflict-resolve <conflict-id> --resolution "以用户最新确认的信息为准"
```

## 当前限制

- 默认 semantic search 使用本地 hash embedding，适合 MVP 和测试，不等价于真实语义模型。
- 会话摘要第一版是规则候选提取，不会自动判断所有隐含事实。
- 冲突检测只覆盖关键词和实体交集，不做深层逻辑推理。
- 项目索引使用正则提取符号，不是完整 AST。
- Python 文件使用标准库 AST 提取基础代码结构，但仍不是完整 LSP 调用图。
- Code Impact Pack 第一版基于索引摘要、符号和关键词重合，不等价于完整调用图。
- `hippo mcp` 是轻量 JSON-RPC stdio 服务，还不是完整 MCP SDK 实现。
- `hippo daemon` 是本地 HTTP 服务入口，还不是 Windows 服务/托盘程序。
- Chroma 和 LLM summarizer 是可选能力，默认仍保持离线可用。
- 暂无完整 GUI、权限系统和云同步。

## 路线图

- Global daemon / tray app
- MCP server
- `hippo run` vibe coding wrapper
- LLM-based summarizer
- AST-based code graph
- Git history memory
- Visual memory browser
- Permission system
- Memory evaluation benchmark
