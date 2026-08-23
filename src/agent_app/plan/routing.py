from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ExecutionMode = Literal["react", "plan_only", "plan_and_execute"]

_CHINESE_ACTION_PATTERN = re.compile(
    r"(?:读取|搜索|检查|分析|定位|发现|实现|修改|修复|重构|迁移|添加|删除|运行|执行|"
    r"测试|验证|构建|安装|配置|授权|整理|写入|创建|调用|完成)"
)
_ENGLISH_ACTION_PATTERN = re.compile(
    r"\b(?:read|search|inspect|analy[sz]e|locate|discover|implement|modify|update|fix|"
    r"refactor|migrate|add|remove|run|execute|test|verify|build|install|configure|"
    r"authorize|organize|write|create|call|complete)\b"
)
_CHINESE_CHANGE_PATTERN = re.compile(r"(?:实现|修改|修复|重构|迁移|添加|删除|写入|创建)")
_CHINESE_VERIFICATION_PATTERN = re.compile(r"(?:运行|执行|测试|验证|构建)")
_ENGLISH_CHANGE_PATTERN = re.compile(
    r"\b(?:implement|modify|update|fix|refactor|migrate|add|remove|write|create)\b"
)
_ENGLISH_VERIFICATION_PATTERN = re.compile(r"\b(?:run|execute|test|verify|build)\b")
_STRUCTURED_STEP_PATTERN = re.compile(
    r"^(?:[-+*]|\d+[.)、]|[一二三四五六七八九十]+[、.])\s*"
)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    mode: ExecutionMode
    goal: str
    explicit: bool
    reason: str


def route_request(user_input: str) -> RouteDecision:
    """Choose ReAct or planning without changing the default simple-task path."""

    stripped = user_input.strip()
    if not stripped:
        raise ValueError("Task input cannot be empty.")
    command, _, raw_goal = stripped.partition(" ")
    lowered_command = command.casefold()
    goal = raw_goal.strip()
    if lowered_command == "/plan":
        if not goal:
            raise ValueError("/plan requires a goal.")
        return RouteDecision("plan_only", goal, True, "explicit_plan_command")
    if lowered_command in {"/plan-and-execute", "/plan_execute"}:
        if not goal:
            raise ValueError("/plan-and-execute requires a goal.")
        return RouteDecision("plan_and_execute", goal, True, "explicit_plan_and_execute_command")
    if lowered_command in {"/react", "/reactive"}:
        if not goal:
            raise ValueError("/react requires a goal.")
        return RouteDecision("react", goal, True, "explicit_react_command")
    multi_step_reason = _multi_step_reason(stripped)
    if multi_step_reason is not None:
        return RouteDecision("plan_and_execute", stripped, False, multi_step_reason)
    return RouteDecision("react", stripped, False, "simple_or_exploratory_task")


def _multi_step_reason(text: str) -> str | None:
    """Return a high-confidence Plan signal, never a single keyword match."""

    lowered = text.casefold()
    if _looks_like_question(lowered):
        return None
    if _has_structured_action_list(lowered):
        return "structured_step_list"
    if _has_ordered_action_sequence(lowered):
        return "ordered_action_sequence"
    if _has_connected_action_clauses(lowered):
        return "multiple_action_clauses"
    return None


def _looks_like_question(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith(("?", "？")):
        return True
    normalized = stripped.rstrip("。.!！?？ ")
    if normalized.startswith(
        ("请问", "是否", "是不是", "为什么", "怎么", "如何", "也就是说", "能否", "可否", "可不可以")
    ):
        return True
    if normalized.startswith(
        (
            "is ",
            "are ",
            "do ",
            "does ",
            "did ",
            "should ",
            "would ",
            "could ",
            "can ",
            "why ",
            "how ",
            "what ",
            "when ",
            "where ",
            "explain ",
            "tell me ",
        )
    ):
        return True
    return normalized.endswith(("吗", "呢", "对吗", "对吧", "是不是"))


def _has_structured_action_list(text: str) -> bool:
    action_items = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _STRUCTURED_STEP_PATTERN.match(line) is None:
            continue
        item = _STRUCTURED_STEP_PATTERN.sub("", line, count=1)
        if _action_count(item) > 0:
            action_items += 1
    return action_items >= 2


def _has_ordered_action_sequence(text: str) -> bool:
    if _action_count(text) < 2:
        return False
    chinese_sequence = re.search(
        r"(?:首先|第一步|先).+?(?:然后|下一步|其次|再|第二步|最后).+",
        text,
        flags=re.DOTALL,
    )
    english_sequence = re.search(
        r"\b(?:first|firstly)\b.+?\b(?:then|next|second|finally)\b.+",
        text,
        flags=re.DOTALL,
    )
    return chinese_sequence is not None or english_sequence is not None


def _has_connected_action_clauses(text: str) -> bool:
    if _action_count(text) < 2:
        return False
    if re.search(r"(?:然后|再|最后)", text) or re.search(
        r"\b(?:and then|then|after that|next|finally)\b", text
    ):
        return True
    chinese_change_then_verify = re.search(
        rf"{_CHINESE_CHANGE_PATTERN.pattern}.{{0,80}}(?:并且|同时|以及|并).{{0,80}}"
        rf"{_CHINESE_VERIFICATION_PATTERN.pattern}",
        text,
    )
    english_change_then_verify = re.search(
        rf"{_ENGLISH_CHANGE_PATTERN.pattern}.{{0,80}}\band\b.{{0,80}}"
        rf"{_ENGLISH_VERIFICATION_PATTERN.pattern}",
        text,
    )
    return chinese_change_then_verify is not None or english_change_then_verify is not None


def _action_count(text: str) -> int:
    return len(_CHINESE_ACTION_PATTERN.findall(text)) + len(
        _ENGLISH_ACTION_PATTERN.findall(text)
    )
