# Agent Harness 现状评估与改造计划

## 一、文档目的

这份文档基于当前仓库实际代码重写，目标是回答三个问题：

1. 现在这个项目已经具备了哪些 agent harness 能力
2. 目前真正缺的是什么，优先级应该怎么排
3. 后续应该按什么顺序改造，避免过早堆叠“看起来很高级”的能力

当前评估范围以 `src/agent_app/` 和 `tests/` 为准，结论针对的是**单 Agent、CLI 形态、面向 coding agent 方向的 MVP harness**，不是多 Agent 平台，也不是 Web 产品后端。

---

## 二、当前项目的准确定位

当前项目不是“只有一个 loop 的 demo”，也不是“生产级 agent 平台”。

更准确的定位是：

- 一个已经成型的 **single-agent CLI harness**
- 具备最小可用的 **prompt + tool loop + session persistence + safety guard**
- 已经有 **单元测试和集成测试**
- 但离真正好用的 coding agent 还有几层关键抽象没有补上

如果和 `learn-claude-code`、`cc-haha` 这类项目相比，你现在已经有了最底层骨架，但还没有形成 Claude Code 风格那种“可连续工作、可安全改代码、可累积上下文、可稳定验证”的完整闭环。

---

## 三、当前已经具备的 Harness 能力

### 3.1 Agent 定义层

当前已经有明确的 agent 定义对象，而不是把 prompt 和工具白名单散落在各处。

- `AgentDefinition` 已抽象出 `goal`、`system_prompt_template`、`rules`、`allowed_tools`、`default_model`、`max_tool_rounds`
- 当前主 agent 已限制工具集合为 `file_read`、`code_search`、`file_write`、`shell`
- prompt 中已经包含一些 coding-agent 规则，例如：
  - 优先直接回答
  - 位置类问题优先 `code_search` / `file_read`
  - 写文件后优先做最小验证
  - 验证失败不自动回滚

这说明项目已经进入“有 agent policy”的阶段，而不只是 API 调用封装。

### 3.2 Orchestrator / 主循环

当前 `AgentLoop` 已经覆盖了单 agent harness 最核心的执行闭环：

- 读取 session
- 追加用户消息
- 渲染 system prompt
- 调用模型
- 执行 tool call
- 把 tool result 回填给模型
- 返回最终回答

除此之外，还具备几项已经比较像正式 harness 的行为：

- `max_tool_rounds` 上限，避免无限循环
- 同一工具连续失败 2 次后停止当前 turn
- 在某些问题类型下，能基于已有证据提前回答，避免无意义继续调工具
- tool 调用顺序按模型返回顺序串行执行

注意：
这里的“连续失败后停止”更接近 **turn 内失败保护**，还不能算完整 circuit breaker。

### 3.3 Tool Plane / 工具平面

当前工具层已经有基础安全边界，不是裸奔状态。

已具备：

- `ToolRegistry` 做工具注册和 schema 暴露
- `file_read` 有 workspace 路径约束
- `code_search` 优先使用 `rg`，没有时退回 Python 实现
- `file_write` 具备写入前检查
- `shell` 仅允许白名单命令，且限制为 PowerShell 兼容形式
- `file_write` 默认需要人工确认
- CLI 确认时会显示 diff 摘要或内容预览

`file_write` 的保护已经不算弱，包含：

- 限制目标路径必须在 workspace 内
- 禁写 `.git`、`.agent_app`、`.env.local` 等敏感位置
- 限制可写文件后缀
- 限制最大字节数和行数
- 已存在文件如果太大，不允许整文件覆盖

这部分是当前项目一个比较扎实的基础。

### 3.4 Session / 状态持久化

当前项目已经不是无状态 CLI。

已具备：

- SQLite 持久化 `sessions`、`messages`、`tool_runs`
- session 复用
- `current_session.txt` 记录最近 session
- CLI 支持：
  - 指定 `--session-id`
  - `--new-session`
  - 默认复用最近 session

这让项目已经具备了“跨轮对话”的最低能力。

### 3.5 Model Adapter / 模型适配层

当前已经单独封装了 OpenAI-compatible client，而不是把 HTTP 调用塞进 loop 里。

已具备：

- 统一的 `generate()` 接口
- 组装 `system + messages + tools`
- 解析文本回答
- 解析 tool calls
- 基本错误分类：
  - `configuration_error`
  - `http_error`
  - `request_error`
  - `invalid_json`
  - `invalid_response`
  - `invalid_tool_arguments`

这意味着后面切换 provider 或补 tracing 时有明确落点。

### 3.6 测试基础

当前项目已经有比较像样的测试骨架，而不是“只跑过手工 demo”。

已具备：

- unit tests
- integration tests
- 对以下模块有覆盖：
  - agent definition / prompt
  - config
  - db
  - session service
  - tools
  - model adapter
  - orchestrator loop
  - CLI flow

这点很重要，因为后面改 harness 时，测试会是你防止能力退化的第一道保障。

---

## 四、当前真实缺口

这一节只写“当前项目真实缺什么”，不把未来 Web 产品能力提前混进来。

### 4.1 上下文工程还很初级

这是当前最核心的缺口之一。

现在的 session memory 主要是：

- 每轮从数据库取最近 16 条 `messages`
- 把这些消息直接传给模型

当前缺失：

- 没有 token budget 管理
- 没有滚动摘要
- 没有把上一轮 `tool_runs` 重新组织为后续可用证据
- 没有 scratchpad / todo memory
- 新 session 完全没有长期记忆

最关键的不是“还没上向量库”，而是：

**跨轮的工具证据没有被重新利用。**

也就是说，虽然 `tool_runs` 已经入库，但它们目前更像日志，而不是 agent 的可回忆工作记忆。

### 4.2 编辑能力太弱，只能 whole-file write

这会很快成为 coding agent 的上限。

当前 `file_write` 只能：

- 创建小文件
- 覆盖小文件

缺失：

- patch / diff 式编辑
- replace-in-file
- 精准局部修改
- 写后自动读取目标片段再验证

这意味着只要文件略大、修改略复杂，agent 很容易卡在“知道怎么改，但没有安全好用的改法”。

从 coding agent 角度看，这一项优先级高于长期记忆和 Web API。

### 4.3 配置维度不够，model/tool runtime 混在一起

当前只有一个 `timeout` 配置，同时给模型请求和 shell 执行使用。

这会带来几个问题：

- 模型超时和 shell 超时的合理值不同
- 后续如果新增 `code_search_timeout`、`file_read_limit`、`verification_timeout`，配置会越来越乱
- 出问题时很难判断是模型太慢还是工具太慢

因此这里不是“有 bug”，而是**配置抽象还没长出来**。

### 4.4 工具参数校验仍然分散在各工具内部

当前每个工具都自己写一套：

- `isinstance`
- 判空
- 数值范围
- 错误文案

这在现阶段够用，但后面工具一多会出现：

- 校验风格不一致
- 错误格式不一致
- schema 和实际校验可能漂移

所以“统一 schema 校验”是值得做的，但它属于第二阶段的工程升级，不是眼下第一刀。

### 4.5 工程可运行性还有一个明显短板

当前测试虽然是绿的，但默认执行体验不够顺。

现状：

- 运行 `python -m unittest discover -s tests -v` 会因为找不到 `agent_app` 失败
- 需要先设置 `PYTHONPATH=src`

这说明项目还没有做到：

- 开箱即测
- 可安装包结构
- 明确统一的测试入口

对个人手搓项目来说，这其实是很值得尽早补的一层工程基础。

### 4.6 还没有 runtime abstraction

当前 shell 调用是直接 `subprocess.run(...)`。

这对 CLI MVP 是够的，但后面一旦你想做这些能力，就会吃力：

- 可取消执行
- 更细粒度的执行状态
- 多命令编排
- 独立 verification runtime
- 沙箱隔离
- 后台任务 / 长任务管理

所以问题不在于“现在就一定要上容器”，而在于：

**shell 现在还是工具，不是 runtime。**

### 4.7 缺少评测与可观测性

当前有测试，但没有形成 agent 评测闭环。

缺失：

- 任务级 benchmark
- 回归 case 集合
- 关键行为指标
- 结构化 trace / turn log

如果以后你持续增强上下文工程、tool policy、编辑工具，没有这层评测，很容易出现“能力看起来更强了，实际稳定性反而下降”。

---

## 五、上一版分析里哪些判断需要纠偏

上一版文档方向大体是对的，但有几处需要修正。

### 5.1 “已实现简单熔断机制”表述偏大

当前更准确的说法应为：

- 有 **turn 内连续失败保护**
- 还没有完整的 circuit breaker

因为目前没有：

- 冷却期
- 跨 turn 状态
- 跨 session 状态
- 全局限流

### 5.2 “子进程僵尸”不是当前最紧急问题

当前使用的是 `subprocess.run(..., timeout=...)`。

在现有 CLI MVP 下，更应该优先关注的是：

- shell 能力抽象太弱
- 无法表达复杂验证动作
- 没有 runtime 级取消与隔离

等你把 shell 从“工具”升级为“runtime”之后，再系统性处理进程组、取消传播、后台任务会更合适。

### 5.3 FastAPI / SSE / 断连感知不是当前第一优先级

这些能力对 Web 产品很重要，但对当前项目的核心问题帮助有限。

在现阶段，更高优先级的其实是：

1. 让测试与包结构顺起来
2. 让上下文跨轮真正可用
3. 让代码编辑能力从 whole-file write 升级为 patch-based
4. 让 timeout / validation / verification 这些核心抽象稳定下来

### 5.4 向量库不是当前第一刀

长期记忆当然有价值，但在现阶段你更缺的是：

- turn 内和跨 turn 的工作记忆
- 可压缩的 tool evidence
- token budget
- scratchpad / todo

如果这些还没做好，先接 ChromaDB 很可能只是把复杂度提前引入。

---

## 六、推荐的改造顺序

下面这份路线图的原则是：

- 先补最影响开发效率和正确性的能力
- 再补上下文工程
- 最后再补平台化和高级能力

---

## Phase 0：工程基线收口

目标：
先把项目变成“开箱能跑、开箱能测、可持续迭代”的状态。

### 要做什么

1. 统一测试入口
   - 让仓库根目录直接执行测试成功
   - 不再依赖手工设置 `PYTHONPATH=src`

2. 补项目打包/安装元信息
   - 增加 `pyproject.toml`
   - 明确包路径和测试入口

3. 补 coverage 门槛
   - 核心业务代码单测覆盖率目标设为 `>= 90%`
   - 将集成测试命令固定下来

4. 固化开发命令
   - 本地测试命令
   - 运行 CLI 命令
   - 可选：CI 命令

### 完成标志

- 直接在仓库根目录执行测试命令可以通过
- 不需要手工设置 `PYTHONPATH`
- 覆盖率门槛有落地位置

### 建议测试

- package/import 测试
- 全量 unit tests
- CLI integration tests

---

## Phase 1：先把 Harness 核心抽象补齐

目标：
解决当前最影响可用性的结构问题。

### 要做什么

1. 拆分 timeout 配置
   - `model_timeout`
   - `tool_timeout`
   - 可选：`verification_timeout`

2. 引入统一工具调用结果结构
   - 让 tool result 能区分：
     - 原始输出
     - 给模型看的摘要
     - 给日志/数据库留存的完整内容

3. 补 `file_read` 读取上限
   - 限制单次读取窗口
   - 避免整文件读入带来的风险

4. 统一工具参数校验入口
   - 保持现有 schema
   - 把“schema 声明”和“参数校验”尽量收敛

### 完成标志

- model/tool timeout 不再混用
- `file_read` 有稳定边界
- 各工具校验行为更一致

### 建议测试

- config tests
- tool timeout tests
- file_read 边界测试
- schema/参数错误测试

---

## Phase 2：补 coding agent 真正缺的编辑能力

目标：
让 agent 从“能写小文件”升级到“能安全改现有代码”。

### 要做什么

1. 新增 patch 型编辑工具
   - `replace_in_file` 或 `apply_patch`
   - 支持局部替换
   - 支持失败时给出清晰错误

2. 保留 `file_write`，但退居辅助角色
   - 小文件创建继续用 `file_write`
   - 大多数修改改由 patch 工具承担

3. 补写后验证策略
   - 写后自动读取目标片段
   - 再决定是否调用 shell 做最小验证

4. 强化 CLI 确认信息
   - patch preview
   - 更明确的 diff 展示

### 完成标志

- 常见代码修改不再依赖 whole-file overwrite
- 对中等规模文件可稳定局部编辑

### 建议测试

- replace success/failure tests
- multiple match / no match tests
- patch preview tests
- loop + write + verify integration tests

---

## Phase 3：补真正有用的上下文工程

目标：
让 agent 跨轮不只是“记得聊天”，而是“记得工作证据”。

### 要做什么

1. 引入 token budget
   - 不再固定只取最近 16 条消息
   - 依据 token 预算动态裁剪上下文

2. 增加滚动摘要
   - 老消息压缩成 summary
   - 避免上下文无限膨胀

3. 把 `tool_runs` 升级为可回忆证据
   - 存原始结果
   - 生成模型可消费的摘要
   - 下轮按需注入

4. 增加 scratchpad / todo memory
   - 用于复杂任务分解
   - 明确“当前计划、已完成、待验证”

### 完成标志

- 跨轮对话时能复用历史工具证据
- 长对话不会简单退化成“只记最近 16 条消息”

### 建议测试

- message budget tests
- summary generation tests
- cross-turn memory tests
- tool evidence retrieval tests

---

## Phase 4：补 runtime 与评测闭环

目标：
让项目从“可用的 CLI harness”升级到“可以稳定迭代的 agent 系统”。

### 要做什么

1. 抽 runtime 层
   - 把 shell 从普通工具升级成运行时能力
   - 为后续取消、隔离、验证、后台任务做准备

2. 增加 tracing / turn log
   - 记录每轮：
     - prompt 组装
     - tool call
     - tool result
     - stop reason

3. 建立任务级评测
   - 选一批固定任务
   - 检查：
     - 是否调用了合理工具
     - 是否写对文件
     - 是否通过验证

4. 固化回归测试集
   - 每次改上下文工程或 tool policy 都回归跑一遍

### 完成标志

- 能稳定判断改造是否带来回归
- runtime 能承载更复杂的执行模型

### 建议测试

- runtime abstraction tests
- trace structure tests
- task-level regression tests

---

## Phase 5：高级能力，最后再上

这一阶段才建议考虑：

- subagents
- 长期记忆 / 向量检索
- Web API / SSE
- 更复杂的多用户并发
- 更重的数据库或异步框架

原因很简单：

如果前面几层没打稳，这些能力大概率只会增加复杂度，而不会明显提高当前 agent 的稳定性。

---

## 七、不建议当前优先投入的方向

以下方向不是不能做，而是当前不应该排到前面。

### 7.1 FastAPI + SSE

这是产品形态升级，不是当前 harness 核心短板。

### 7.2 ChromaDB / 向量记忆

在跨轮工作记忆、摘要和 token budget 没做好前，收益有限。

### 7.3 `aiosqlite`

当前 SQLite 还不是瓶颈。

### 7.4 文件锁

目前是单进程 CLI，`file_write` 的首要问题不是并发，而是编辑粒度太粗。

---

## 八、建议的近期执行清单

如果从今天开始按最合理顺序做，我建议先按下面 6 个小里程碑推进：

1. 让测试命令开箱可跑
2. 拆分 `model_timeout` 和 `tool_timeout`
3. 给 `file_read` 加读取窗口上限
4. 新增 patch 型编辑工具
5. 让 `tool_runs` 变成跨轮可复用证据
6. 加 token budget + 滚动摘要

这 6 步做完，你的项目会从“有 agent 骨架”明显升级到“有实用价值的 coding agent harness”。

---

## 九、阶段目标总结

一句话总结当前项目的现实情况：

你已经搭好了一个不错的单 agent CLI harness 地基。

下一步最值得做的，不是马上平台化，也不是先追多 agent 和向量库，而是：

- 补工程基线
- 升级编辑能力
- 做好跨轮工作记忆
- 建立评测与验证闭环

只要这几层按顺序补齐，后面无论你继续向 Claude Code 风格靠，还是往 OpenHands 风格平台演进，都会顺很多。
