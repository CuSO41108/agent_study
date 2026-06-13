# Agent Study

Agent Study 是一个面向本地仓库工作流的 CLI-first coding agent harness。它把模型调用、工具执行、文件编辑、session 持久化、任务记忆、子代理委派和 trace 记录放在一个可测试的 Python 项目里，用来验证 agent 如何在本地代码仓库中可靠工作。

## 项目简介

这个项目关注的是 coding agent 的执行骨架，而不是应用 UI 或托管服务。当前主线已经可以完成一轮或多轮本地仓库任务：读取代码、搜索文件、维护 todo、局部修改文件、执行受限 shell 验证，并把 session 与执行证据保存到工作区下的 SQLite 数据库。

相比从教学 demo 出发的 agent 项目，Agent Study 更关注本地执行、状态持久化、工具安全边界和可回归验证。

## 适用场景

- 研究 coding agent 的 tool-call loop、上下文组装和停止条件。
- 验证本地文件读写、搜索、shell 执行和人工确认的安全边界。
- 观察跨轮 session memory、summary、todo 和 recent tool evidence replay 对任务连续性的影响。
- 作为小型 coding agent runtime / harness 的实验基座。

## 当前能力

- CLI 单轮执行：每次命令处理一个用户请求并返回 JSON 结果。
- 交互式 REPL：同一进程内持续对话，支持自然语言确认与 `:new` 新建 session。
- Task persistence：每个用户目标都有独立 `TaskState`，生命周期、计划、工作记忆、预算、PendingAction、Observation 和终止原因写入 SQLite。
- Unified Event：用户输入、批准、拒绝、暂停、恢复、取消和过期都归一化为 append-only `AgentEvent`，使用任务内 sequence 与 version 防止重复或过期迁移。
- 工作记忆：通过 rolling summary、Task Plan 和 recent tool evidence replay 复用当前 session 的上下文；旧 session todo 会在首次建 Task 时导入。
- 安全工具层：提供 `file_read`、`code_search`、`replace_in_file`、`file_write`、`shell`、`todo_read`、`todo_write` 和 `delegate_task`。
- 局部编辑：现有文件优先通过 `replace_in_file` 修改，`file_write` 主要用于小文件创建或显式覆盖。
- 受限 shell：shell 命令经过白名单与参数检查，并通过共享 `ShellRuntime` 执行。
- Bounded delegation：root coordinator 可以把边界清晰的子任务委派给 worker agent。
- ReAct Observe：工具结果、校验失败、拒绝、超时、冲突和执行异常统一转换为语义化 `Observation`；只读或幂等瞬时失败最多自动重试两次。
- 可恢复审批：文件编辑先持久化 PendingAction 并进入 `waiting_user`，可在新进程中批准或拒绝，批准前会复核文件版本。
- 任务预算：限制模型调用、工具调用、token、活跃时间、重复决策、重试和 Reflection 重规划次数。
- 完整 Trace：模型、决策、审批、工具尝试、Observation、预算和状态迁移即时关联到 `task_id`；兼容的 turn/tool 摘要仍写入 `turn_traces` 和 `tool_call_traces`。

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

查询和控制持久化任务：

```powershell
python -m agent_app.cli --workspace-root . --task-status TASK_ID
python -m agent_app.cli --workspace-root . --pause-task TASK_ID
python -m agent_app.cli --workspace-root . --resume-task TASK_ID
python -m agent_app.cli --workspace-root . --cancel-task TASK_ID
python -m agent_app.cli --workspace-root . --approve-task TASK_ID
python -m agent_app.cli --workspace-root . --reject-task TASK_ID
```

交互模式会让同一个进程持续运行，便于继续回答模型的自然语言追问。输入 `:new` 可以开始新 session，输入 `exit` 或 `quit` 退出。

`--workspace-root` 应指向真实存在的工作区根目录。CLI 会把 `.agent_app/agent.db` 和 `.agent_app/current_session.txt` 存在该目录下，也会从同一路径加载 `.agent_app/.env.local`。

## 架构地图

- `src/agent_app/cli.py`：CLI 入口、session 解析与交互循环。
- `src/agent_app/agent/`：agent 定义、目标、规则和工具访问策略。
- `src/agent_app/orchestrator/`：ReAct 主循环、Event 入口、Executor、context builder 和 subagent runner。
- `src/agent_app/tools/`：文件、代码搜索、shell、todo、delegate 与编辑工具。
- `src/agent_app/state/`：SQLite session、Task 快照、append-only Event、ToolAction 和 Trace 持久化。
- `src/agent_app/runtime/`：`TaskRuntime` 状态机与共享 `ShellRuntime`。
- `tests/`：单元、集成和回归测试，其中包含 `tests/regression/`。

## 设计边界

- 当前形态是本地 CLI harness；服务化、Web UI 和托管运行不是主线。
- 编排层由项目内 tool-call loop 实现，没有引入 LangChain / LangGraph 等外部编排框架。
- Planner、Critic、Reflection 仅按条件轻量触发，不构建独立常驻决策流水线。
- `waiting_tool` 只保留状态和迁移契约，本期工具仍为同步执行。
- Governance 目前只保留工具副作用、幂等性和风险元数据，不包含 RBAC、租户、告警或外部指标平台。
- 当前上下文策略以 summary、todo 和 recent tool evidence replay 为主；更重的上下文检索机制先不放入 `main`。
- `replace_in_file` 不是通用 patch 系统，不支持模糊匹配、正则匹配或多文件编辑。
- 当前 shell 体验以 PowerShell 为中心。

## 路线图

1. 稳定 CLI 使用体验：继续打磨安装、配置、错误提示和交互式流程。
2. 强化工具安全边界：补齐文件编辑、路径校验、shell 白名单和人工确认的边缘场景。
3. 补强评测闭环：沉淀更多固定任务，验证工具调用、文件修改和 trace 记录是否符合预期。
4. 改进任务连续性：围绕现有 summary、todo 和 evidence replay 优化跨轮上下文。
5. 优化子代理委派：继续收紧 worker agent 的边界、结果摘要和失败处理。
6. 根据真实任务需要，再评估是否增加更明确的规划/反思步骤或更重的上下文检索机制。

## 测试

运行标准测试集：

```powershell
python -m unittest discover -s tests -v
```

运行核心模块覆盖率：

```powershell
python -m coverage run -m unittest discover -s tests -v
python -m coverage report --fail-under=90
python -m coverage report --precision=2 --fail-under=90
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

- TaskState 状态机架构：[`docs/TASK_STATE_MACHINE.md`](docs/TASK_STATE_MACHINE.md)
- 历史架构评估：[`docs/AGENT_CONTEXT_ENGINEERING.md`](docs/AGENT_CONTEXT_ENGINEERING.md)
- 兼容启动器：`run_local.ps1` 仍然是 `agent-app` 的薄包装，不注入 `PYTHONPATH` 或模型密钥。
