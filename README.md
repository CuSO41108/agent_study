# Agent Study

Agent Study 是一个面向本地仓库工作流的 CLI 优先 coding agent harness。它的目标不是一开始就做成完整平台，也不是复刻某个成熟产品，而是把一个可以真实执行工具、修改文件、记录上下文、做最小验证的 agent 工程骨架打扎实。

当前 `main` 保留的是轻量主线：单进程 agent loop、session 持久化、todo / summary 工作记忆、安全文件工具、受限 shell、局部编辑、子代理委派和 trace。2A / 2B / 2C 的 retrieval 实验不进入 `main`。

## 项目定位

这个项目更准确地说是：

- 一个 CLI-first coding agent harness。
- 一个用来研究“agent 如何在本地仓库里可靠工作”的工程实验。
- 一个可以连续运行、读写文件、调用工具、保存 session 和验证结果的最小系统。
- 不是 Web 服务，不是托管平台，也不是完整多 agent 产品。

和 MokioAgent / MokioClaw 那种“从最小 ToolCall 开始，沿着清晰迭代路线长成 Mini Claw”的方向相比，Agent Study 当前更偏底层 harness：它已经有较多运行时、安全、状态和测试基础，但缺少一个更明确的教学任务主线。后续更适合围绕一个具体任务场景继续迭代，而不是过早增加知识检索系统。

## 当前没有使用什么

当前项目没有使用 LangChain 或 LangGraph，`pyproject.toml` 里也没有相关依赖。

当前也没有实现严格意义上的 Plan-and-Execute 或 ReAct 框架：

- 没有独立的 planner / executor / reviewer 图结构。
- 没有 LangGraph 状态机。
- 没有 LangChain agent executor。
- 现在是自写的 tool-call loop：模型根据 system prompt 决定是否调用工具，loop 执行工具，把结果回填给模型，直到得到最终回答或触发停止条件。

它有一些类似 ReAct 的行为，例如“看问题 -> 调工具 -> 观察结果 -> 再回答”，但这只是 tool-call loop 的自然形态，不是显式引入 ReAct 框架。

## Retrieval 复盘

当初加 2A / 2B / 2C retrieval 的动机是合理的，但更像后续实验，不是当前主线的必需能力。

2A：Session Evidence Retrieval  
目标是从当前 session 里召回已经完成的工作证据，包括成功的 `tool_runs`、session summary、以及成功 turn 的结论，减少 agent 重复查同样的信息。

2B：Docs RAG  
目标是从 README 和少量白名单文档中召回说明性内容，让 agent 更容易回答“怎么用、怎么配置、项目架构是什么”这类问题。

2C：Code-Aware Retrieval  
目标是对代码位置、符号定义、模块归属和实现行为类问题，在模型调用工具前先注入一些候选代码证据。

现在决定不把它们放进 `main`，原因是：

- 当前项目还没有清晰到必须依赖 retrieval 的任务场景。
- `code_search` + `file_read` 已经能覆盖很多仓库定位问题，而且证据更直接。
- 已有的 rolling summary、todo scratchpad、recent tool evidence replay 已经足够支撑 MVP 级上下文工程。
- Docs RAG 和 code-aware retrieval 会引入额外抽象、测试和调参成本，容易让项目主线从“agent harness”变成“检索系统”。

所以 `main` 的判断是：先保证 agent 可以完整运转，再考虑是否在单独分支重新设计 retrieval。

## 当前能力

- 通过 CLI 执行单轮任务，也可以启动交互式 REPL。
- 默认复用最近一次本地 session，也支持显式新建 session。
- 使用 SQLite 保存 messages、tool runs、session summary、subagent runs 和 traces。
- 使用 `todo_read` / `todo_write` 维护当前任务计划。
- 使用 `file_read`、`code_search`、`replace_in_file`、`file_write`、`shell` 等工具完成仓库内操作。
- 对文件路径、隐藏文件、敏感目录、shell 命令和写入大小做安全限制。
- 对现有文件优先使用 `replace_in_file` 做局部修改。
- 支持 root coordinator 通过 `delegate_task` 委派清晰子任务给 worker agent。
- 记录 turn 级和 tool 级 trace，并写入 `turn_traces` / `tool_call_traces`，便于复盘行为。

## 快速开始

在仓库根目录安装项目和开发依赖：

```powershell
python -m pip install -e .[dev]
```

查看 CLI 帮助：

```powershell
agent-app --help
python -m agent_app.cli --help
```

准备本地配置：

```powershell
New-Item -ItemType Directory -Force .agent_app | Out-Null
Copy-Item .env.example .agent_app\.env.local
```

然后打开 `./.agent_app/.env.local`，填入模型提供方配置：

```dotenv
MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_API_KEY=your_api_key_here
MODEL_NAME=qwen3.6-plus
MODEL_TIMEOUT=30
TOOL_TIMEOUT=30
```

- `MODEL_API_KEY`：你的模型提供方 API key。
- `MODEL_NAME`：提供方接受的模型标识符。
- `MODEL_BASE_URL`：OpenAI-compatible 接口地址。上面的示例指向 DashScope compatible mode。
- `MODEL_TIMEOUT`：模型请求超时时间。
- `TOOL_TIMEOUT`：工具执行超时时间，未设置时回退到 `MODEL_TIMEOUT`。
- `CONTEXT_TOKEN_BUDGET`：单轮上下文的大致 token 预算。
- `SUMMARY_TRIGGER_TOKENS`：何时把更早对话压缩成 session summary。

如果只想在当前 PowerShell 会话里临时覆盖模型配置：

```powershell
$env:MODEL_API_KEY = "your_api_key_here"
$env:MODEL_NAME = "qwen3.6-plus"
```

执行过 `python -m pip install -e .[dev]` 之后，一般不需要再手动设置 `PYTHONPATH`。

## 基本使用

执行单轮请求：

```powershell
python -m agent_app.cli "hello, what model are you" --workspace-root .
```

默认复用最近一次本地 session：

```powershell
python -m agent_app.cli "do you remember my last question" --workspace-root .
```

显式开启新 session：

```powershell
python -m agent_app.cli "hello" --workspace-root . --new-session
```

启动交互式 REPL：

```powershell
python -m agent_app.cli --interactive --workspace-root .
agent-app --interactive --workspace-root .
```

交互模式会让同一个进程持续运行，便于继续回答模型的自然语言追问。输入 `:new` 可以开始新 session，输入 `exit` 或 `quit` 退出。

`--workspace-root` 应指向真实存在的工作区根目录。CLI 会把 `.agent_app/agent.db` 和 `.agent_app/current_session.txt` 存在该目录下，也会从同一路径加载 `.agent_app/.env.local`。

## 模块地图

- `src/agent_app/cli.py`：CLI 入口、session 解析与交互循环。
- `src/agent_app/agent/`：agent 定义、目标、规则和工具访问策略。
- `src/agent_app/orchestrator/`：主循环、context builder 和 subagent runner。
- `src/agent_app/tools/`：文件、代码搜索、shell、todo、delegate 与编辑工具。
- `src/agent_app/state/`：基于 SQLite 的 session 存储、summary、subagent run 和 traces。
- `src/agent_app/runtime/`：共享 `ShellRuntime`，统一 shell 执行与验证路径。
- `tests/`：单元、集成和回归测试，其中包含 `tests/regression/`。

## 当前边界

- 这是本地 coding workflow 的 MVP harness，不是完整平台。
- 还没有正式的 planner / executor / reviewer 执行图。
- 还没有基于 LangChain / LangGraph 的 orchestration。
- 没有长期记忆或跨项目知识库。
- `replace_in_file` 不是通用 patch 系统，不支持模糊匹配、正则匹配或多文件编辑。
- 当前 shell 体验以 PowerShell 为中心。

## 后续路线

1. 保持当前 main 轻量：稳定 CLI、tool loop、session、编辑和验证闭环。
2. 选一个明确任务主线，例如 Mini Game Studio 或小型代码维护任务集，用它驱动后续能力取舍。
3. 增加 Reflection：让 agent 对生成或修改后的代码做自检，并基于验证结果给出修复建议。
4. 评估是否需要显式 Plan-and-Execute：只有当任务复杂到单个 tool-call loop 不好控制时，再引入 planner / executor / reviewer。
5. 继续补强 MultiAgent，但保持 bounded delegation，不急着做通用多 agent 平台。
6. 深化 runtime / harness：取消、隔离、审批、trace、回放和回归评测。
7. Retrieval 暂时放到实验分支。只有当任务场景证明 `code_search`、`file_read`、summary 和 todo 不够用时，再重新设计。

## 测试与验证

运行标准测试集：

```powershell
python -m unittest discover -s tests -v
```

运行核心模块覆盖率：

```powershell
python -m coverage run -m unittest discover -s tests -v
python -m coverage report --fail-under=90
```

快速 smoke test：

```powershell
python -m agent_app.cli --interactive --workspace-root .
```

仓库定位类验证可以使用：

```powershell
python -m agent_app.cli --workspace-root . "当前项目哪里定义了 session 相关逻辑？请给出文件路径。"
```

做 repo-aware 验证时，优先使用带有项目符号名或文件名的提示词。检查返回 JSON，确认 `tool_runs` 中出现了 `code_search` 或 `file_read`，不要只看答案像不像对。

session persistence demo：

```powershell
python -m agent_app.cli --workspace-root . "我最喜欢的数字是 42"
python -m agent_app.cli --workspace-root . "还记得我刚才说最喜欢的数字是什么吗"
python -m agent_app.cli --workspace-root . "还记得我刚才说最喜欢的数字是什么吗" --new-session
```

第二条命令应复用前一个 `session_id` 并记住 `42`；带 `--new-session` 的命令应创建不同的 `session_id`，不再带上之前上下文。

## 延伸文档

- 架构与演进说明：[`docs/AGENT_CONTEXT_ENGINEERING.md`](docs/AGENT_CONTEXT_ENGINEERING.md)
- 兼容启动器：`run_local.ps1` 仍然是 `agent-app` 的薄包装，不注入 `PYTHONPATH` 或模型密钥。
