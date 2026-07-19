# Feature Specification: Controlled Shell Approval

**Feature Branch**: `003-controlled-shell-approval`  
**Created**: 2026-07-11  
**Status**: Draft

## User Scenarios & Testing

### User Story 1 - Approve bounded workspace shell changes (Priority: P1)

用户要求创建生成物目录、移动或复制工作区内文件时，Agent 展示规范化命令、风险理由和受影响路径；用户批准后才执行。

**Independent Test**: 请求创建 `outputs/` 并移动一个文件；拒绝时文件不变，批准时仅目标路径发生预期变化。

### User Story 2 - Preserve approval and deletion boundaries (Priority: P2)

用户或模型尝试递归/批量删除时，系统明确拒绝且不执行。其他 Shell 命令，包括未分类命令和组合命令，必须在人工明确批准后才执行。

**Independent Test**: 对递归删除确认硬拒绝；对任意其他命令确认未经批准或人工拒绝时没有副作用，且 trace 记录审批决定。

## Requirements

- **FR-001**: 系统 MUST 拒绝空命令和 AGENTS.md 明确禁止的递归/批量删除形式；其他 Shell 命令进入人工审批，不以预置命令白名单限制 Coding Agent 能力。
- **FR-002**: 系统 MUST 将每个 Shell 命令置入既有 `tool_approval` 生命周期，除非用户已为当前 Session 明确授权匹配的命令前缀。
- **FR-003**: 经人工批准的 Shell 命令从 workspace 作为工作目录执行；审批是权限边界，不承诺把任意命令的文件、网络或子进程影响限制在 workspace 内。
- **FR-004**: 系统 MUST 在审批提示与 trace 中记录命令、风险等级、规范化影响路径和结果，但不得记录密钥或敏感环境值。
- **FR-005**: 系统 MUST 保持 `file_write` 和 `replace_in_file` 的现有审批及恢复语义；shell 审批不得绕过其文件校验。
- **FR-006**: 递归删除及项目 AGENTS.md 禁止的删除形式 MUST 继续拒绝。

## Success Criteria

- **SC-001**: 受支持的目录创建、移动和复制在批准前 100% 不产生副作用。
- **SC-002**: 所有硬拒绝、未批准或人工拒绝的 Shell 输入均不执行，且任务 trace 可说明原因与审批决定。
- **SC-003**: 现有文件编辑审批、恢复和 shell 白名单测试保持通过。
