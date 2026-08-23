from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import PurePosixPath


_PRIVATE_FILENAME_TERMS = ("简历", "面经", "面试", "导学")
_PRIVATE_ASCII_FILENAME = re.compile(r"(?:^|[-_.])(resume|interview|cv)(?:[-_.]|$)", re.IGNORECASE)
_PRIVATE_EXTENSIONS = {".db", ".sqlite", ".sqlite3", ".log", ".pem", ".p12", ".pfx"}
_ALLOWED_EMAIL_DOMAINS = {
    "example.com",
    "example.invalid",
    "users.noreply.github.com",
}

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_CHINA_MOBILE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_WINDOWS_USER_PATH = re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE)
_UNIX_USER_PATH = re.compile(r"/(?:Users|home)/[^/\s]+", re.IGNORECASE)
_SECRET_PATTERNS = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("GitHub token", re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("provider API key", re.compile(r"(?:sk-|tvly-)[A-Za-z0-9_-]{20,}")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("JWT", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?m)^\s*([A-Z][A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD))\s*[:=]\s*([^\s#]+)"
)
_SAFE_ASSIGNMENT_MARKERS = (
    "example",
    "invalid",
    "placeholder",
    "replace",
    "your_",
    "your-",
    "test",
    "dummy",
    "secrets.",
    "${",
)


def scan_path(path: str) -> list[str]:
    normalized = path.replace("\\", "/")
    name = PurePosixPath(normalized).name
    folded_name = name.casefold()
    findings: list[str] = []

    if any(term in name for term in _PRIVATE_FILENAME_TERMS) or _PRIVATE_ASCII_FILENAME.search(folded_name):
        findings.append("private career/study filename")
    if folded_name.startswith(".env") and folded_name != ".env.example":
        findings.append("local environment file")
    if PurePosixPath(normalized).suffix.casefold() in _PRIVATE_EXTENSIONS:
        findings.append("private runtime/credential file extension")
    if normalized.casefold().startswith(".agent_app/"):
        findings.append("AgentLab local runtime state")
    return findings


def scan_content(path: str, content: bytes) -> list[str]:
    if b"\0" in content:
        return []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return []

    findings: list[str] = []
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(label)

    for match in _SECRET_ASSIGNMENT.finditer(text):
        value = match.group(2).strip("\"'").casefold()
        if len(value) >= 12 and not any(marker in value for marker in _SAFE_ASSIGNMENT_MARKERS):
            findings.append(f"literal credential assignment ({match.group(1)})")

    personal_domains = {
        match.group(1).casefold()
        for match in _EMAIL.finditer(text)
        if match.group(1).casefold() not in _ALLOWED_EMAIL_DOMAINS
    }
    if personal_domains:
        findings.append("non-example email address")
    if _CHINA_MOBILE.search(text):
        findings.append("Chinese mobile number")
    if _WINDOWS_USER_PATH.search(text) or _UNIX_USER_PATH.search(text):
        findings.append("absolute user-home path")
    return findings


def _git_paths(*, staged: bool) -> list[str]:
    command = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
        if staged
        else ["git", "ls-files", "-z"]
    )
    completed = subprocess.run(command, check=True, capture_output=True)
    return [item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]


def _index_content(path: str) -> bytes:
    completed = subprocess.run(["git", "show", f":{path}"], check=True, capture_output=True)
    return completed.stdout


def check_repository(*, staged: bool) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for path in _git_paths(staged=staged):
        for reason in scan_path(path):
            findings.append((path, reason))
        content = _index_content(path)
        for reason in scan_content(path, content):
            findings.append((path, reason))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject tracked files with common privacy or secret risks.")
    parser.add_argument("--staged", action="store_true", help="Check only staged added or modified files.")
    args = parser.parse_args(argv)

    findings = check_repository(staged=args.staged)
    if not findings:
        scope = "staged changes" if args.staged else "tracked repository"
        print(f"Privacy check passed for {scope}.")
        return 0

    print("Privacy check failed; sensitive values are intentionally not displayed:", file=sys.stderr)
    for path, reason in sorted(set(findings)):
        print(f"  {path}: {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
