# Agent Study

一个面向本地仓库工作流的 CLI-first coding agent harness——把模型调用、工具执行、文件编辑、session 持久化、任务状态机和 trace 记录放在可测试的 Python 项目里，验证 agent 如何在本地代码仓库中可靠工作。

相比教学 demo，这个项目更关注本地执行、状态持久化、工具安全边界和可回归验证。

## 快速开始

```powershell
# 1. 安装
python -m pip install -e .[dev]

# 2. 配置模型
New-Item -ItemType Directory -Force .agent_app | Out-Null
Copy-Item .env.example .agent_app\.env.local
# 编辑 .agent_app/.env.local，填入你的 MODEL_BASE_URL、MODEL_API_KEY、MODEL_NAME
```

项目兼容 **OpenAI Chat Completions 协议**，任何实现 `/v1/chat/completions` 接口的模型提供方均可使用（阿里云百炼、OpenAI、DeepSeek、Ollama 等）。详见 `.env.example` 中的注释。

```powershell
# 3. 运行第一条命令
agent-app "src/agent_app/state/ 目录下有哪些文件，各自负责什么"
```

## 核心亮点

**TaskState 持久化状态机** — 每个用户目标都有独立的生命周期（created → running → waiting_user → completed/failed/cancelled），状态迁移以事务和乐观锁保证一致性。任务可以在一个进程中暂停、在另一个进程中恢复审批，所有中间状态落 SQLite。

**受控 Shell 审批** — 只读白名单命令（git status、dir、Get-Content 等）自动放行；目录创建、文件移动和复制进入审批流程并追踪受影响路径；递归删除、管道组合、越界路径硬拒绝，拒绝原因写入 trace。

**副作用安全与崩溃恢复** — 文件编辑和 Shell 变更执行前持久化 ToolAction 与幂等键。有副作用的操作崩溃后不自动重试，而是要求用户重新审批，避免写入重复或不确定状态。

**结构化观察与预算控制** — 工具执行结果、超时、冲突、拒绝统一为结构化 Observation，附带错误类型和可重试标记。模型调用、工具调用、token、活跃时间、重试次数和重复决策均设上限，任一超限即终止。

**Web Search 集成** — 用户显式要求检索公开信息时（"查阅"/"搜索网页"等），自动触发 web search 预研，结果按来源 URL 注入模型上下文，确保可溯源。

**Eval 评测闭环** — 20 个固定任务覆盖文件编辑、shell 边界、工具预算等场景。支持 dry-run 和 live-model 两种模式，保留工作区快照，输出 JSONL 报告。

**完整 Trace 可观测** — 模型调用、审批决策、工具执行、预算快照和状态迁移按 task_id 即时关联，支持跨进程回溯问题根因。247 项测试，核心代码覆盖率 90%。

## 日常使用

```powershell
agent-app                    # 进入交互式 REPL
agent-app "帮我分析项目结构"   # 单轮执行
agent-app "hello" --new-session  # 开启新 session
```

**REPL 命令：**

| 命令 | 说明 |
|---|---|
| `/task` | 查看当前 session 最新 task |
| `/tasks` | 列出当前 session 所有 task |
| `/approve [task-id前缀]` | 批准待审批的工具动作 |
| `/reject [task-id前缀]` | 拒绝待审批的工具动作 |
| `/cancel [task-id前缀]` | 取消非终态 task |
| `/new` | 新建 session |
| `/help` | 查看帮助 |
| `exit` | 退出 |

**跨进程 task 控制：**

```powershell
agent-app --task-status TASK_ID     # 查看 task 状态
agent-app --approve-task TASK_ID    # 批准
agent-app --reject-task TASK_ID     # 拒绝
agent-app --cancel-task TASK_ID     # 取消
```

## 测试

```powershell
python -m unittest discover -s tests -v         # 全量测试
python -m coverage run -m unittest discover -s tests -v
python -m coverage report --precision=2 --fail-under=90
```

## 设计边界

- 本地 CLI harness，不做服务化、Web UI 或托管运行
- ReAct 循环由项目内 `AgentLoop` 实现，不引入 LangChain/LangGraph 等外部编排框架
- 工具路径约束在工作区内；shell 以 PowerShell 为中心
- `replace_in_file` 不是通用 patch 系统，不支持模糊匹配、正则或多文件编辑
- `waiting_tool` 保留状态契约但工具仍为同步执行
- 上下文策略以 summary、todo 和 recent tool evidence replay 为主，不做重型上下文检索
- 不设 RBAC、租户、告警等 governance 机制

## 延伸文档

- [架构地图](docs/ARCHITECTURE.md)
- [TaskState 状态机](docs/TASK_STATE_MACHINE.md)
- [路线图](docs/ROADMAP.md)
- [Eval 使用指南](docs/EVAL_DEMO.md)
