# Feature Specification: Web Research Tool

**Feature Branch**: `002-web-search`  
**Created**: 2026-07-11  
**Status**: Draft  
**Input**: User description: "增加 web_search 工具及其 API 配置；要求查阅时先检索，并澄清由 ReAct 自主决定工具调用。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Research current public information (Priority: P1)

用户要求“查阅”“联网检索”或明确要求搜索公开资料时，Agent 先取得检索结果和来源，再基于这些资料回答或继续完成任务。

**Why this priority**: 这使“查阅”成为可验证的事实获取，而不是模型用已有记忆生成的内容。

**Independent Test**: 配置有效检索服务后，执行一个明确要求查阅的请求；结果包含至少一项来源，任务 trace 显示成功的检索动作发生在最终回答之前。

**Acceptance Scenarios**:

1. **Given** 已配置可用检索服务，**When** 用户要求查阅某个公开主题，**Then** 系统在生成最终事实性回答前执行一次检索，并将结果和来源提供给后续决策。
2. **Given** 已配置可用检索服务，**When** 用户要求查阅并创建一个页面，**Then** 系统先检索资料，再由后续决策选择是否创建文件及如何验证。

---

### User Story 2 - Receive an actionable unavailable-search result (Priority: P2)

用户请求查阅，但检索服务未配置、凭据无效、限流或不可达时，能得到明确失败原因，而不会把模型记忆伪装成查阅结果。

**Why this priority**: 用户需要区分“事实未找到”和“系统没有完成检索”，并能据此修复配置。

**Independent Test**: 在未配置检索服务的工作区执行“查阅”请求，任务失败信息指出检索不可用，且没有生成声称来源于检索的最终回答。

**Acceptance Scenarios**:

1. **Given** 未配置检索服务，**When** 用户明确要求查阅，**Then** 系统以配置不足的可识别原因结束任务。
2. **Given** 检索服务返回错误，**When** 用户明确要求查阅，**Then** trace 保存可分类的失败语义和脱敏诊断信息。

---

### User Story 3 - Configure and observe web research (Priority: P3)

操作者可以通过工作区配置启用检索服务，并在任务记录中查看一次检索所耗时间、请求摘要、结果数量、来源链接和错误类别，不泄露密钥。

**Why this priority**: 可配置性和可观测性使该能力能在不同本地环境中安全运行和排障。

**Independent Test**: 使用示例配置启动 CLI 并执行查阅请求，检查持久化 trace；其中不包含密钥、完整认证头或未脱敏凭据。

**Acceptance Scenarios**:

1. **Given** 配置了检索服务，**When** Agent 执行检索，**Then** trace 记录检索动作、耗时、成功/失败和来源数量。
2. **Given** 配置值无效，**When** CLI 启动或首次检索，**Then** 系统给出可操作的配置错误，而非笼统的模型错误。

### Edge Cases

- 用户只要求根据已有文本生成内容，而未请求查阅时，不触发检索。
- 检索响应缺少可用来源时，任务不得把该响应表述为已查阅的可靠资料。
- 返回内容过长或来源过多时，只向后续模型决策提供有界、可追溯的摘要。
- 检索服务超时、返回非预期格式或拒绝请求时，错误分类需与模型调用错误区分。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统 MUST 提供一个仅用于公开网络检索的 `web_search` 能力，并将其作为 Agent 可选择的工具公开。
- **FR-002**: 系统 MUST 允许操作者通过工作区配置提供检索服务地址、认证凭据、超时和结果数量限制；所有凭据 MUST 保持脱敏。
- **FR-003**: 当用户明确要求“查阅”“联网检索”或等义的当前公开资料检索时，系统 MUST 在最终事实性回答前完成至少一次 `web_search`；若该动作不可完成，系统 MUST 明确失败而不得用模型记忆替代。
- **FR-004**: 除 FR-003 的明确检索要求外，ReAct 决策组件 MUST 继续自主决定是否以及何时调用可用工具；本功能不得引入独立常驻 planner、supervisor 或多 Agent 工作流。
- **FR-005**: 系统 MUST 将检索的结构化结果和来源作为观察结果交给后续 ReAct 决策，以便决定后续回答、文件创建和验证动作。
- **FR-006**: 系统 MUST 为每次检索持久化关联任务的可观测记录，至少包括动作类型、耗时、结果或来源数量，以及可分类的错误原因；记录不得包含密钥或认证头。
- **FR-007**: 系统 MUST 将检索配置错误、网络/超时错误、服务拒绝、非预期响应和无来源结果与 `model_error` 区分。
- **FR-008**: 系统 MUST 保持现有 CLI、任务持久化、审批和本地工具的兼容行为。

### Key Entities *(include if feature involves data)*

- **Search Configuration**: 控制检索服务连接、认证、时间限制和结果上限的工作区配置。
- **Search Result**: 一次公开网络检索返回的标题、摘要、来源链接和可选发布日期。
- **Search Observation**: 供任务状态和后续 ReAct 决策使用的有界检索结果或分类失败。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 在有效配置下，100% 的明确“查阅”测试请求在最终回答前记录至少一次成功检索动作和一个来源链接。
- **SC-002**: 在缺失或无效配置下，100% 的明确“查阅”测试请求在一次任务内返回可识别的检索错误，且不生成伪称已查阅的最终答案。
- **SC-003**: 每次检索的可观测记录均可通过 task ID 关联，且抽样记录中 0 个包含认证凭据。
- **SC-004**: 现有不要求查阅的 CLI、持久化和本地工具测试保持通过。

## Assumptions

- 第一版使用一个可通过 HTTPS 调用的公开网页检索服务；操作者自行提供有效凭据。
- “查阅”规则只覆盖用户明确的检索意图，不把普通的创作、改写和已有文本总结自动扩展为联网请求。
- 原有单 coordinator 的 ReAct 架构保留；对明确检索要求的强制前置动作属于运行时策略，而不是新增的多 Agent planner。
