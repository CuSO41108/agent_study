from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Literal


SkillScope = Literal["project", "user"]

_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_MAX_SKILL_BYTES = 64 * 1024
_MAX_RESOURCE_BYTES = 64 * 1024
_MAX_INDEX_DESCRIPTION = 240


@dataclass(frozen=True, slots=True)
class SkillSummary:
    name: str
    description: str
    scope: SkillScope
    source_path: Path
    version: str | None = None
    platforms: tuple[str, ...] = ()
    requires_tools: tuple[str, ...] = ()
    invocation: str | None = None


@dataclass(frozen=True, slots=True)
class SkillManifest:
    name: str
    description: str
    version: str | None = None
    platforms: tuple[str, ...] = ()
    requires_tools: tuple[str, ...] = ()
    invocation: str | None = None


@dataclass(frozen=True, slots=True)
class SkillDocument:
    summary: SkillSummary
    content: str
    content_hash: str
    directory: Path


@dataclass(frozen=True, slots=True)
class SkillResource:
    skill_name: str
    relative_path: str
    content: str
    content_hash: str


class SkillRegistry:
    """Discover Skills from project and user roots without ever mutating them."""

    def __init__(self, workspace_root: str | Path, *, user_root: str | Path | None = None) -> None:
        self._workspace_root = Path(workspace_root).resolve()
        self._project_root = self._workspace_root / "skills"
        self._user_root = (
            Path(user_root).expanduser().resolve()
            if user_root is not None
            else Path.home() / ".agent-study" / "skills"
        )
        self._warnings: tuple[str, ...] = ()

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def user_root(self) -> Path:
        return self._user_root

    @property
    def warnings(self) -> tuple[str, ...]:
        self.discover()
        return self._warnings

    def discover(self) -> tuple[SkillSummary, ...]:
        """Return the compact index; project entries shadow user entries by name."""
        warnings: list[str] = []
        entries: dict[str, SkillSummary] = {}
        for scope, root in (("user", self._user_root), ("project", self._project_root)):
            for summary in self._discover_root(root=root, scope=scope, warnings=warnings):
                entries[summary.name] = summary
        self._warnings = tuple(warnings)
        return tuple(entries[name] for name in sorted(entries))

    def resolve(self, name: str) -> SkillSummary | None:
        normalized = _validate_skill_name(name)
        if normalized is None:
            return None
        return next((item for item in self.discover() if item.name == normalized), None)

    def load(self, name: str) -> SkillDocument | None:
        summary = self.resolve(name)
        if summary is None:
            return None
        try:
            content = _read_text_limited(summary.source_path, max_bytes=_MAX_SKILL_BYTES)
        except (OSError, UnicodeError, ValueError):
            return None
        return SkillDocument(
            summary=summary,
            content=content,
            content_hash=_content_hash(content),
            directory=summary.source_path.parent,
        )

    def load_active(
        self,
        *,
        name: str,
        scope: str,
        source_path: str,
        content_hash: str,
    ) -> tuple[SkillDocument | None, str | None]:
        """Load a persisted activation only when its origin and hash still match."""
        document = self.load(name)
        if document is None:
            return None, "skill is no longer available"
        if document.summary.scope != scope:
            return None, "Skill scope changed"
        if str(document.summary.source_path) != source_path:
            return None, "Skill source changed"
        if document.content_hash != content_hash:
            return None, "Skill content changed"
        return document, None

    def load_resource(self, name: str, relative_path: str) -> SkillResource | None:
        """Read a file explicitly referenced by an already-loaded Skill body."""
        document = self.load(name)
        if document is None:
            return None
        return self.load_resource_from_document(document, relative_path)

    def load_resource_from_document(self, document: SkillDocument, relative_path: str) -> SkillResource | None:
        """Read a level-2 resource from a previously hash-verified Skill document."""
        normalized = _safe_relative_path(relative_path)
        if normalized is None or not _body_mentions_resource(document.content, normalized):
            return None
        target = (document.directory / normalized).resolve()
        if not _is_within(target, document.directory) or target == document.summary.source_path:
            return None
        try:
            content = _read_text_limited(target, max_bytes=_MAX_RESOURCE_BYTES)
        except (OSError, UnicodeError, ValueError):
            return None
        return SkillResource(
            skill_name=document.summary.name,
            relative_path=normalized,
            content=content,
            content_hash=_content_hash(content),
        )

    def target_path(self, *, scope: SkillScope, skill_name: str) -> Path:
        normalized = _validate_skill_name(skill_name)
        if normalized is None:
            raise ValueError("Skill name must be a lower-case slug.")
        root = self._root_for_scope(scope)
        target = root / normalized / "SKILL.md"
        if not _is_within(target, root):
            raise ValueError("Skill target escaped its configured root.")
        return target

    def create_new_skill(self, *, scope: SkillScope, content: str) -> Path:
        """Create a validated new Skill only; existing Skills are never overwritten."""
        manifest = parse_skill_manifest(content)
        if self.resolve(manifest.name) is not None:
            raise FileExistsError(f"A Skill named '{manifest.name}' already exists in a configured root.")
        target = self.target_path(scope=scope, skill_name=manifest.name)
        target_directory = target.parent
        if target_directory.exists():
            raise FileExistsError(f"Skill directory already exists: {target_directory}")
        root = self._root_for_scope(scope)
        root.mkdir(parents=True, exist_ok=True)
        target_directory.mkdir()
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        return target

    def model_index(self, *, max_skills: int = 24) -> str | None:
        all_entries = self.discover()
        entries = all_entries[:max_skills]
        if not entries:
            return None
        lines = [
            "Available read-only Skills (metadata only; load one only when its description matches the task):"
        ]
        for item in entries:
            details = [f"scope={item.scope}"]
            if item.invocation:
                details.append(f"invocation={item.invocation}")
            if item.platforms:
                details.append("platforms=" + ",".join(item.platforms))
            lines.append(
                f"- {item.name}: {_compact(item.description, _MAX_INDEX_DESCRIPTION)} "
                f"({'; '.join(details)})"
            )
        if len(all_entries) > len(entries):
            lines.append("- More Skills exist; call skill_list to inspect the full index.")
        lines.append("Use skill_load(name) to activate and read a matching Skill. Skills cannot be edited by the agent.")
        return "\n".join(lines)

    def _discover_root(
        self,
        *,
        root: Path,
        scope: SkillScope,
        warnings: list[str],
    ) -> tuple[SkillSummary, ...]:
        if not root.is_dir():
            return ()
        results: list[SkillSummary] = []
        try:
            candidates = sorted(root.glob("*/SKILL.md"), key=lambda item: item.parent.name.casefold())
        except OSError as exc:
            warnings.append(f"Unable to scan {root}: {exc}")
            return ()
        for manifest in candidates:
            try:
                resolved_manifest = manifest.resolve()
                if not _is_within(resolved_manifest, root.resolve()):
                    warnings.append(f"Ignored Skill outside configured root: {manifest}")
                    continue
                content = _read_text_limited(resolved_manifest, max_bytes=_MAX_SKILL_BYTES)
                metadata = _parse_frontmatter(content)
                name = _validate_skill_name(metadata.get("name"))
                description = metadata.get("description", "").strip()
                if name is None or not description:
                    raise ValueError("frontmatter requires safe 'name' and non-empty 'description'")
                if name != manifest.parent.name:
                    raise ValueError("frontmatter name must match the Skill directory")
                results.append(
                    SkillSummary(
                        name=name,
                        description=description,
                        scope=scope,
                        source_path=resolved_manifest,
                        version=_optional_scalar(metadata.get("version")),
                        platforms=_csv_values(metadata.get("platforms")),
                        requires_tools=_csv_values(metadata.get("requires_tools")),
                        invocation=_optional_scalar(metadata.get("invocation")),
                    )
                )
            except (OSError, UnicodeError, ValueError) as exc:
                warnings.append(f"Ignored invalid Skill '{manifest.parent.name}': {exc}")
        return tuple(results)

    def _root_for_scope(self, scope: SkillScope) -> Path:
        if scope == "project":
            return self._project_root
        if scope == "user":
            return self._user_root
        raise ValueError("Skill scope must be project or user.")


def parse_skill_manifest(content: str) -> SkillManifest:
    if len(content.encode("utf-8")) > _MAX_SKILL_BYTES:
        raise ValueError(f"Skill content exceeds {_MAX_SKILL_BYTES} byte limit")
    metadata = _parse_frontmatter(content)
    name = _validate_skill_name(metadata.get("name"))
    description = metadata.get("description", "").strip()
    if name is None or not description:
        raise ValueError("frontmatter requires safe 'name' and non-empty 'description'")
    return SkillManifest(
        name=name,
        description=description,
        version=_optional_scalar(metadata.get("version")),
        platforms=_csv_values(metadata.get("platforms")),
        requires_tools=_csv_values(metadata.get("requires_tools")),
        invocation=_optional_scalar(metadata.get("invocation")),
    )


def _parse_frontmatter(content: str) -> dict[str, str]:
    if not content.startswith("---\n") and not content.startswith("---\r\n"):
        raise ValueError("missing YAML frontmatter")
    lines = content.splitlines()
    metadata: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return metadata
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line or line.startswith((" ", "\t")):
            raise ValueError("frontmatter must use simple key: value fields")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key or not value or key in metadata:
            raise ValueError("invalid or duplicate frontmatter field")
        metadata[key] = _unquote(value)
    raise ValueError("frontmatter is not closed")


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _validate_skill_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    return normalized if _SKILL_NAME.fullmatch(normalized) else None


def _optional_scalar(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _csv_values(value: object) -> tuple[str, ...]:
    scalar = _optional_scalar(value)
    if scalar is None:
        return ()
    raw = scalar.strip("[]")
    return tuple(part.strip().strip("'\"") for part in raw.split(",") if part.strip())


def _safe_relative_path(value: str) -> str | None:
    canonical = value.replace("\\", "/")
    candidate = Path(canonical)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        return None
    normalized = candidate.as_posix()
    return normalized if normalized not in {".", "SKILL.md"} else None


def _body_mentions_resource(content: str, relative_path: str) -> bool:
    body = content.split("\n---", 1)[-1]
    return relative_path in body or f"./{relative_path}" in body


def _read_text_limited(path: Path, *, max_bytes: int) -> str:
    if not path.is_file():
        raise ValueError("not a regular file")
    if path.stat().st_size > max_bytes:
        raise ValueError(f"file exceeds {max_bytes} byte limit")
    return path.read_text(encoding="utf-8")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _content_hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _compact(value: str, limit: int) -> str:
    compacted = " ".join(value.split())
    return compacted if len(compacted) <= limit else compacted[: limit - 1] + "…"
