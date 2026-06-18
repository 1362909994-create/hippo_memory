# hippocampus-memory 项目报告

日期：2026-06-18

## 1. 执行摘要

`hippocampus-memory` 是一个本地优先的 AI coding 外部记忆系统。它的核心价值不是做普通 RAG，而是把项目长期信息压缩成 AI 可以直接使用的工作上下文：Memory Pack、Project Profile、Code Map、Code Impact Pack 和 Context Bundle。

当前项目已经达到“可在 Reasonix 上试用”的阶段：可以一键部署、自动注入上下文、通过轻量 MCP 自动召回和存储记忆，并在 Reasonix 状态栏显示会话级预计 token 节省。它还没有达到“稳定产品”的阶段：真实 UI 端到端测试不足，token 节省仍是估算，不是模型账单实测；Reasonix bundle patch 属于脆弱集成，Reasonix 升级后可能失效。

我的判断：这个项目有意义，但目标用户应收窄到“长期使用 AI coding CLI 的重度开发者/小团队”。如果面向泛用户，它会太复杂；如果面向 AI coding workflow，它有真实需求。

## 2. 目前进度

已经完成：

- 本地 SQLite 记忆库。
- 记忆类型、可见性、置信度、重要性、软删除/硬删除。
- 项目配置 `.hippo.toml` 和项目自动识别。
- 项目索引：文件摘要、hash、chunks、imports、symbols、calls。
- Python AST 优先索引，其他语言正则 fallback。
- Memory Pack、Project Profile、Code Map、Code Impact Pack、Context Bundle。
- 自动召回策略 `auto-context`。
- 自动存储策略 `auto-store`，支持 candidate queue。
- 冲突检测和显式解决流程。
- token-report 和 token-ledger。
- 轻量 JSON-RPC stdio MCP。
- Reasonix 一键部署、全局 shim、状态栏补丁、会话级预计节省显示。
- Reasonix 卸载入口：CLI 和 PowerShell 脚本。
- 单元测试和集成式 CLI 测试。

未完成或不稳定：

- 没有真实 Reasonix UI 端到端自动化测试。
- 没有真实模型请求级 token/cost 对账。
- 没有跨机器 fresh install CI。
- 没有规模化 benchmark。
- MCP 还不是完整 MCP SDK 实现。
- 代码图不是完整语言服务器或准确调用图。

## 3. 目前功能和实现方式

### 3.1 本地记忆库

实现位置：

- `hippocampus_memory/db.py`
- `hippocampus_memory/memory_writer.py`
- `hippocampus_memory/retriever.py`
- `hippocampus_memory/packer.py`

实现方式：

- SQLite 存储项目、记忆、文件、chunks、symbols、calls、candidate、conflict、token ledger。
- 记忆有类型、project、visibility、confidence、importance、created_at、updated_at。
- 删除区分 soft delete 和 hard delete。
- 检索使用关键词、hash embedding、可选质量增强，之后做重排和去重。

确认方式：

- `tests/test_db.py` 覆盖数据库初始化、写入、删除、统计。
- `tests/test_retriever.py` 覆盖检索、过滤、排序。
- `tests/test_packer.py` 覆盖 Memory Pack 输出和敏感过滤。

### 3.2 项目索引和代码上下文

实现位置：

- `hippocampus_memory/project_indexer.py`
- `hippocampus_memory/ast_indexer.py`
- `hippocampus_memory/code_map.py`
- `hippocampus_memory/code_intelligence.py`
- `hippocampus_memory/code_graph.py`
- `hippocampus_memory/change_planner.py`

实现方式：

- 遍历项目文件，跳过 `.git`、`.venv`、`node_modules`、build/dist、二进制、大文件等。
- 存储文件路径、hash、摘要、chunks。
- Python 文件用标准库 AST 提取 class/function/import/call。
- 非 Python 用正则 fallback。
- Code Map 输出相关文件和符号。
- Code Impact Pack 根据 intent、记忆、文件摘要、符号和调用线索生成改动影响提示。

确认方式：

- `tests/test_project_indexer.py` 覆盖索引、跳过规则、missing 文件、Python AST。
- `tests/test_context_packs.py` 覆盖 Context Bundle 和影响包。
- `tests/test_lsp_diagnostics.py` 覆盖可选 LSP 诊断回退。

### 3.3 自动召回

实现位置：

- `hippocampus_memory/recall_policy.py`
- `hippocampus_memory/context_bundle.py`
- `hippocampus_memory/callback.py`

实现方式：

- 根据 intent 判断是否需要外部记忆。
- 小闲聊跳过。
- 继续任务返回 callback/compact pack。
- 调试和代码修改返回 lean Context Bundle。
- 项目综述返回 full Context Bundle。
- 显式记忆查询返回 Memory Pack。

确认方式：

- `tests/test_auto_policies.py` 覆盖策略分流、metadata、token savings text。
- `tests/test_cli.py` 覆盖 CLI 输出和 token 账本。

### 3.4 自动存储

实现位置：

- `hippocampus_memory/memory_policy.py`
- `hippocampus_memory/session_ingestor.py`
- `hippocampus_memory/summarizer.py`
- `hippocampus_memory/llm_summarizer.py`

实现方式：

- 从会话摘要或文本中识别长期有价值信息。
- 高置信、非敏感信息直接写入。
- 不确定或敏感内容进入 candidate queue。
- 低价值闲聊、重复日志、临时噪音跳过。
- LLM summarizer 可选，失败后回退规则摘要。

确认方式：

- `tests/test_auto_policies.py` 覆盖自动写入、候选、跳过。
- `tests/test_writer.py` 覆盖写入策略。

### 3.5 Reasonix 集成

实现位置：

- `hippocampus_memory/deploy.py`
- `hippocampus_memory/cli.py`
- `install-reasonix-hippo.ps1`
- `uninstall-reasonix-hippo.ps1`

实现方式：

- `hippo reasonix-deploy` 创建项目本地 `.hippo` 数据库、MCP 启动脚本和项目配置。
- 写入 `~/.reasonix/config.json` 的 MCP 项：`hippo_memory=hippo mcp-project`。
- 全局 `reasonix` shim 在启动 `reasonix code` 时生成 Context Bundle，并通过 `--system-append-file` 注入 Reasonix。
- 状态栏补丁读取 `HIPPO_REASONIX_STATUS_FILE` 指向的 JSON，按 Reasonix session id 记录 ledger。
- 新会话从 0 开始，旧会话按该 session 的 ledger 恢复。
- 只有 Reasonix 有真实对话轮次后，才把本次 Context Bundle 估算节省计入会话。
- 卸载入口恢复原始 shim 和 Reasonix bundle，移除 MCP 配置和全局提示块。

确认方式：

- `tests/test_reasonix_deploy.py` 覆盖 deploy、shim、status patch、status JSON、uninstall。
- 本机实测：已恢复 Reasonix 原始 shim 和 UI bundle，移除 `hippo_memory` MCP 项，删除项目 `.hippo/.reasonix`，卸载全局 `hippo`。

## 4. Token 节省到底是什么

当前 token 节省不是账单实测，而是估算：

```text
saved_tokens = baseline_tokens - output_tokens
```

其中：

- `baseline_tokens`：项目记忆和文件摘要如果朴素塞进上下文，大概需要多少 token。
- `output_tokens`：实际生成的 Context Bundle 大概需要多少 token。
- `saved_tokens`：两者差值。

例如之前 UI 里的 `46.9K` 来自：

```text
49984 baseline tokens - 3097 Context Bundle tokens = 46887 tokens
```

这只能证明“压缩包比朴素上下文短”，不能证明“DeepSeek 账单真实少花了 46887 token”。要得到真实账单节省，必须拿到 Reasonix 每轮真实请求和一个“不使用 Hippo 的反事实请求”进行对照；项目目前没有这个权限，也不应该随便拦截 Reasonix 内部请求。

## 5. 做过哪些测试

本轮修改前后运行过：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_reasonix_deploy.py tests/test_install_script.py
.\.venv\Scripts\python.exe -m ruff check hippocampus_memory tests
```

结果：

- Reasonix 部署/卸载相关测试：通过。
- lint：通过。

完整测试将在本报告提交前再次运行。

测试目的：

- 确认 Reasonix 一键部署不会破坏现有配置。
- 确认 shim 可恢复，避免用户卸载后 Reasonix 仍被 Hippo 劫持。
- 确认状态栏 patch 可恢复，避免 Reasonix bundle 长期残留自定义代码。
- 确认项目本地 `.hippo/.reasonix` 可以按参数删除。
- 确认安装和卸载脚本存在并调用正确入口。

## 6. 还没有做的测试

必须补：

- 真实 Reasonix 端到端 UI 测试：启动 Reasonix、输入两轮消息、截图验证状态栏从 0 开始并按 session 隔离。
- Reasonix 升级后兼容测试：bundle 文件名、StatusRow 结构、statusBar anchor 变化时应失败并给出明确错误。
- fresh Windows 测试：干净机器只有 Reasonix、Python、PowerShell，运行安装脚本后能自动使用。
- 卸载幂等测试：安装、卸载、再次卸载、再次安装都不损坏 Reasonix。
- 长会话测试：同一 session 多次打开、resume/new 切换、多个项目并行。
- 隐私泄漏测试：private/sensitive 记忆不进入 pack、bundle、browser、MCP 输出。
- 竞争记忆测试：冲突事实不被自动覆盖。
- 性能测试：10k、100k memories 和大型 repo 的索引、召回、pack 延迟。
- token 估算准确性测试：tiktoken 和 fallback 估算差异。

后续测试优先级：

1. Reasonix UI E2E。
2. 隐私/敏感记忆 admission gate。
3. 长会话 session ledger。
4. fresh install/uninstall。
5. 大规模 benchmark。

## 7. 已经修改不了或不应该强改的东西

### 7.1 Reasonix 内部请求和真实账单

项目现在无法精确知道 Reasonix 发送给模型的完整 prompt 和真实计费细节。强行拦截会带来几个后果：

- 需要 monkey patch Reasonix 网络层，升级极易失效。
- 可能暴露 API key、用户 prompt 或敏感项目内容。
- 会让 Hippo 从“上下文生成器”变成“代理/中间人”，安全边界完全变化。

因此当前只能显示“预计节省”，不能显示“真实账单节省”。

### 7.2 Reasonix 状态栏 patch 的稳定性

状态栏 patch 修改的是 Reasonix 打包后的 JS bundle。能用，但脆弱。Reasonix 改 `StatusRow`、`formatTokens` 或 bundle 结构后，patch 会失效。

强行继续深度 patch 的后果：

- 每次 Reasonix 更新都可能要跟着修。
- 用户机器上的 Reasonix bundle 可能损坏。
- 很难做跨版本质量保证。

更好的方向是推动 Reasonix 官方提供插件位、status bar extension 或 MCP-driven UI API。

### 7.3 完整代码调用图

当前 Code Graph 只是基于索引的轻量调用线索。要做到接近 IDE/LSP 级别，需要语言服务器、类型推断、跨语言依赖解析。强行在本项目里自研会吞掉大量时间，且质量未必超过现成工具。

## 8. 项目冗余

### 8.1 非 Reasonix 通用入口过多

冗余点：

- `hippo run` 支持 print/file/env/stdin/arg 多种注入模式。
- API server、daemon script、browser、MCP、Reasonix shim 同时存在。

当时为什么这样设计：

- 早期不确定最终接哪个 AI CLI，所以保留了多种接入方式。
- 想证明核心上下文包可以被不同宿主使用。

问题：

- 产品焦点变散。
- 测试矩阵变大。
- README 变复杂。

改善：

- 短期 README 主推 Reasonix。
- CLI 保留但标为 advanced。
- API/browser/daemon 暂不继续产品化。

### 8.2 自研轻量代码图和外部 CodeGraph 重叠

冗余点：

- 项目里有 `code_graph.py`、`code_intelligence.py`。
- 开发环境里又有 CodeGraph MCP。

当时为什么这样设计：

- 项目需要无外部依赖、离线可用的基础代码索引。

改善：

- 保留轻量索引作为 fallback。
- 如果用户环境有 CodeGraph，优先使用 CodeGraph 做结构化分析。
- 不再投入大量精力自研完整调用图。

### 8.3 token ledger 和 Reasonix session ledger 概念重叠

冗余点：

- 项目级 `token_ledger` 记录 Context Bundle 估算。
- Reasonix session ledger 记录 UI 会话显示。

当时为什么这样设计：

- 项目级报告和 UI 会话展示需求不同。

问题：

- 用户容易把“项目累计估算”和“当前会话估算”混起来。

改善：

- README 明确区分。
- UI 只显示会话级预计节省。
- 项目级累计只在 `hippo token-ledger` 里展示。

## 9. 高级决策视角：该砍什么

应该砍或冻结：

- 冻结非 Reasonix 的产品化投入。理由：当前需求来自 Reasonix 自动集成，泛 CLI 会分散资源。
- 冻结 browser UI。理由：本地 HTML 报告有用，但不是核心闭环。
- 冻结 daemon/server 产品化。理由：没有权限、身份、同步、服务管理前，HTTP server 容易变成安全负担。
- 冻结自研完整代码图。理由：投入高、回报慢，可用 CodeGraph/LSP 替代。
- 砍掉 README 里的过多 advanced 命令。理由：降低新用户理解成本。

应该保留：

- 本地 SQLite。
- Memory Pack / Project Profile / Impact Pack / Context Bundle。
- 自动存储和自动召回。
- Reasonix 一键安装/卸载。
- token-report，但必须坚持“估算”口径。

## 10. 高级决策视角：必须加什么

必须加：

- Reasonix 官方扩展点或更稳定的 adapter 层。如果官方没有扩展点，至少要把 bundle patch 做成版本探测和失败回滚。
- Memory admission gate。召回不是越多越好，错误记忆进入上下文会改变 agent 行为。
- E2E 测试。这个项目的价值发生在 CLI UI 和真实会话里，只靠单元测试不够。
- benchmark。至少要有中文 AI coding 长会话基准、失败经验避免基准、敏感泄漏基准。
- 明确数据模型版本和迁移策略。长期记忆系统最怕 schema 变动破坏历史数据。
- 可观测性。需要能解释某条记忆为什么被召回、为什么被跳过、为什么写入 candidate。

怎么加：

- 新增 `tests/e2e_reasonix/`，用可控假 Reasonix bundle 或 Playwright/terminal capture 验证 UI。
- 新增 `hippo eval-memory-admission`，输入 JSONL，检查 expected/forbidden recalls。
- 新增 `memory_decisions` 表或扩展 metadata，记录 recall/write 的理由和分数。
- 新增 `hippo doctor`，检查 Reasonix shim、bundle patch、MCP 配置、Python 包、项目 DB。

## 11. 同行对比

参考资料：

- Mem0 文档称其是通用、自改进的 LLM 应用记忆层，并提供平台和开源栈：[docs.mem0.ai](https://docs.mem0.ai/introduction)。
- Mem0 论文报告了相对 full-context 的显著 token/latency 优势：[arXiv:2504.19413](https://arxiv.org/abs/2504.19413)。
- Zep 文档强调企业级 agent memory、temporal knowledge graph、Context Graph/Context Lake 和低延迟检索：[help.getzep.com](https://help.getzep.com/overview)。
- Zep 论文强调 temporal knowledge graph 和 enterprise memory benchmark：[arXiv:2501.13956](https://arxiv.org/abs/2501.13956)。
- Letta 文档展示了 memory blocks、archival memory、shared memory、agent runtime 和 Letta Code：[docs.letta.com](https://docs.letta.com/)。
- PROJECTMEM 是更接近本项目方向的 local-first AI coding memory / governance layer：[arXiv:2606.12329](https://arxiv.org/abs/2606.12329)。

他们做得好的地方：

- Mem0：更像产品和基础设施，重视自动抽取、更新、评测和多框架集成。
- Zep：图模型和时间关系更强，企业场景更明确。
- Letta：把 agent runtime、memory、tools、files、permissions 做成完整平台。
- PROJECTMEM：目标非常聚焦 AI coding，强调 append-only log、provenance 和 pre-action gate。

我们当时没有想到或没有做充分的地方：

- 没有把“记忆准入”当作第一优先级安全边界。
- 没有做 append-only event log，因此审计和回放能力弱于 PROJECTMEM。
- 没有 temporal graph，因此跨时间事实、实体关系和冲突处理比较粗。
- 没有正式 benchmark，因此只能证明功能存在，不能证明效果优于 baseline。
- 没有官方扩展点，只能 patch Reasonix bundle。

我们做得更好的地方：

- 更贴近 Reasonix 这类 coding CLI 的实际使用路径。
- 本地优先，默认无重依赖，Windows 可直接落地。
- 产物不是普通 chunk，而是面向 coding agent 的 Context Bundle / Impact Pack。
- 明确把“最小改动方向、风险、不变量、应跑测试”放进上下文。
- 项目级隔离更强，默认不把不同项目记忆混在一起。

我们差的地方：

- 评测体系弱。
- 记忆抽取和合并不够智能。
- 图和时间关系弱。
- UI 集成依赖 patch，稳定性差。
- 文档之前过杂，且曾出现编码损坏。

## 12. 这个项目有没有意义

有意义，但不是大众工具。

真实需求来自这些场景：

- AI coding 会话经常断，上下文反复丢。
- 大项目里 AI 每次都重新读文件，浪费 token 和时间。
- 历史失败经验没有被记住，AI 重复踩坑。
- 用户有稳定偏好、约束和架构决策，希望 AI 自动遵守。
- 团队需要本地、可审计、不上传的 agent memory。

没有意义或需求弱的场景：

- 一次性问答。
- 小脚本。
- 不重复协作的项目。
- 用户不关心本地记忆和上下文成本。
- 已经在使用完整平台型 memory/agent runtime，并且可以接受云服务。

目标人群：

- AI coding 重度个人开发者。
- 用 Reasonix/Codex/Claude Code 做长期项目的人。
- 不想把项目记忆上传云端的小团队。
- 需要审计 AI 决策上下文的工程团队。
- 做多项目切换、嵌入式/硬件/复杂业务代码的开发者。

## 13. 下一步建议

短期：

1. 完成 Reasonix E2E 测试。
2. 加 `hippo doctor`。
3. 建立 memory admission benchmark。
4. 把 README 进一步压缩，避免高级功能吓退新用户。
5. 明确 token savings 的 UI 文案，不再使用“真实节省”暗示。

中期：

1. 设计 append-only event log。
2. 引入实体和时间关系，但不要直接变成重型图数据库。
3. 做 Reasonix 官方插件或稳定扩展点。
4. 给 memory write/recall 增加解释日志。

长期：

1. 形成 AI coding memory benchmark。
2. 和 CodeGraph/LSP 深度集成，避免重复造调用图。
3. 做团队共享但本地可控的同步方案。
4. 支持更稳的权限和敏感信息策略。

## 14. 本轮整理结论

本轮已经做的整理：

- 修复并新增卸载能力。
- 新增 `hippo reasonix-uninstall`。
- 新增 `uninstall-reasonix-hippo.ps1`。
- 修复安装/卸载脚本中 PowerShell 调 Python `-c` 的路径查找问题。
- 重写 README。
- 写入本报告。
- 已从当前电脑删除全局 Hippo/Reasonix 部署。

当前代码仍应保留在 GitHub，因为它已经具备可试用价值。但下一阶段不应继续扩散功能面，应集中补 E2E、benchmark、memory admission 和稳定部署/卸载。
