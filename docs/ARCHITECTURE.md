# 架构地图

## 目录

```
src/agent_app/
├── cli.py                    CLI 入口、session 解析与交互循环
├── config.py                 配置加载（环境变量 → 项目 .env.local → 用户全局 config.toml）
├── types.py                  核心数据类型（Message, TaskState, Observation 等）
├── agent/
│   ├── definition.py          Agent 定义：工具集、规则、角色
│   └── prompts.py             系统提示模板
├── orchestrator/
│   ├── loop.py                ReAct 主循环：模型调用、工具执行、观察处理
│   ├── context_builder.py     上下文组装（summary、todo、evidence replay）
│   └── subagent_runner.py     子代理委派执行
├── tools/
│   ├── base.py                工具基类、Observation 构造、schema 校验
│   ├── approval.py            命令分类与审批决策
│   ├── shell.py               通用 PowerShell 执行与审批边界
│   ├── file_read.py           文件读取
│   ├── file_write.py          文件写入（含检查点恢复）
│   ├── replace_in_file.py     精准文本替换（含检查点恢复）
│   ├── code_search.py         Grep 代码搜索
│   ├── web_search.py          网络搜索（Tavily API）
│   ├── todo.py                Todo 读写
│   ├── delegate_task.py       子任务委派
│   ├── registry.py            工具注册表与工厂函数
│   └── _path_utils.py         工作区路径解析与安全校验
├── state/
│   ├── db.py                  SQLite schema 初始化与迁移
│   └── session_service.py     Session/Task/Event/Action/Trace 持久化
├── runtime/
│   ├── task_runtime.py        TaskState 状态机
│   ├── shell_runtime.py       Shell 生命周期、超时/中断与进程树清理
│   └── agent_runtime.py       Agent 运行时适配
└── model/
    └── openai_compatible.py   OpenAI 兼容模型客户端

tests/
├── unit/        单元测试（25 个文件）
├── integration/ 集成测试
└── regression/  回归测试

evals/
├── runner.py     Eval 执行器
├── scorers.py    打分逻辑
├── cases/        35 个固定 eval 任务（JSON）
├── fixtures/     任务 fixture 工作区
└── results/      JSONL 报告输出
```

## 核心运行流程

```
用户输入 → CLI(argparse) → load_config(环境变量 / 项目配置 / 用户全局配置)
                                 │
    AgentLoop.run_turn()  ◄──────┘
         │
         ├─ context_builder: 组装 messages (summary + todo + evidence)
         ├─ model_client:    调用 LLM
         ├─ approval:        分类工具调用
         │    ├─ allow   → 直接执行
         │    ├─ confirm → 持久化 PendingAction → 等待用户审批
         │    └─ deny    → 拒绝并记录原因
         ├─ tool.execute():  执行工具
         ├─ observation:     结构化观察结果
         ├─ session_service: 持久化 action / trace
         └─ loop:            判断停止条件（预算/完成/失败）
```

## 设计原则

- **不引入外部编排框架**：ReAct 循环由项目内 `AgentLoop` 实现
- **SQLite 单文件持久化**：所有 session、task、trace 存于 `.agent_app/agent.db`
- **乐观锁防冲突**：TaskState 的 `version` 字段保证并发安全
- **副作用追溯**：所有写操作持久化 ToolAction + 幂等键，崩溃后可恢复
- **工具安全边界**：文件路径约束在工作区；Shell 默认审批、session 前缀授权与递归删除硬拒绝；文件编辑使用检查点
