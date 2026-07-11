# Feature Specification: Controlled Shell Approval

**Feature Branch**: `003-controlled-shell-approval`  
**Created**: 2026-07-11  
**Status**: Draft

## User Scenarios & Testing

### User Story 1 - Approve bounded workspace shell changes (Priority: P1)

用户要求创建生成物目录、移动或复制工作区内文件时，Agent 展示规范化命令、风险理由和受影响路径；用户批准后才执行。

**Independent Test**: 请求创建 `outputs/` 并移动一个文件；拒绝时文件不变，批准时仅目标路径发生预期变化。

### User Story 2 - Preserve safe shell boundaries (Priority: P2)

用户或模型尝试递归删除、越过工作区、带组合操作符或未分类命令时，系统明确拒绝且不执行。

**Independent Test**: 对危险命令测试，确认没有副作用且 trace 有拒绝原因。

## Requirements

- **FR-001**: 系统 MUST 将 shell 输入解析为单一、可分类的 PowerShell 命令及结构化参数；不接受组合操作符或无法安全解析的输入。
- **FR-002**: 系统 MUST 将低风险只读命令自动允许，将受限工作区写操作置入既有 `tool_approval` 生命周期。
- **FR-003**: 第一阶段 MUST 支持经审批的目录创建、文件移动和文件复制；所有源和目标路径 MUST 留在 workspace 内。
- **FR-004**: 系统 MUST 在审批提示与 trace 中记录命令、风险等级、规范化影响路径和结果，但不得记录密钥或敏感环境值。
- **FR-005**: 系统 MUST 保持 `file_write` 和 `replace_in_file` 的现有审批及恢复语义；shell 审批不得绕过其文件校验。
- **FR-006**: 递归删除及项目 AGENTS.md 禁止的删除形式 MUST 继续拒绝。

## Success Criteria

- **SC-001**: 受支持的目录创建、移动和复制在批准前 100% 不产生副作用。
- **SC-002**: 所有拒绝的危险 shell 输入均不执行，且任务 trace 可说明原因。
- **SC-003**: 现有文件编辑审批、恢复和 shell 白名单测试保持通过。
