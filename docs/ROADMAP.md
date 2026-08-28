# 路线图

1. 稳定 CLI 使用体验：继续打磨安装、配置、错误提示、任务时间线与交互式流程。
2. 强化工具安全边界：补齐文件编辑、路径校验、通用 Shell 审批、进程中断清理和敏感字段脱敏；Plan 恢复已支持通过不可变人工证据收敛不确定 ToolAction，并按节点完整动作历史阻止混合副作用被误完成或重放。
3. 补强评测闭环：沉淀真实 coding 任务，关联评测结果、失败样本与任务执行记录，形成可复现实验指标；Plan-and-Execute 已补齐 SQLite trace 与端到端验收。
4. 改进任务连续性：在 summary、todo 和 evidence replay 之外，用 SQLite 结构化 Memory 记录任务摘要并支持跨 Session 字面关键词检索；不引入 Embedding 或向量检索。
5. 优化子代理委派：继续收紧 worker agent 的边界、结果摘要和失败处理；仅在独立子任务场景评估并行与文件冲突控制。
6. 已补齐只读 Trace audit replay 与无副作用 dry replay；继续完善 Eval/CI 发布证据、独立 Worker 隔离和更细的 Skill/MCP 运维能力；代码 RAG、Embedding、向量数据库、共享工作区并发和完整 Multi-Agent 不属于当前实现。
