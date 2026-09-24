"""Static repository checks run by CI. Needs only stdlib + PyYAML."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
failures: list[str] = []


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def check_yaml() -> None:
    try:
        import yaml
    except ImportError:
        failures.append("PyYAML is not installed (pip install pyyaml)")
        return

    files = sorted((ROOT / ".github").rglob("*.yml")) + sorted(
        (ROOT / ".github").rglob("*.yaml")
    )
    if not files:
        failures.append("no YAML files found under .github/")
        return

    for path in files:
        name = rel(path)
        try:
            docs = [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8"))]
        except Exception as exc:  # noqa: BLE001 - report any parse error
            failures.append(f"{name}: {exc}")
            continue
        if not any(isinstance(doc, dict) for doc in docs):
            failures.append(f"{name}: no mapping document found")
            continue
        print(f"ok   {name}")


def check_workflows() -> None:
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        name = rel(path)
        text = path.read_text(encoding="utf-8")
        for field in ("name:", "on:", "jobs:"):
            if not re.search(rf"^{re.escape(field)}", text, re.MULTILINE):
                failures.append(f"{name}: missing top-level '{field}'")
        print(f"ok   {name} (structure)")


def check_requirements() -> None:
    path = ROOT / "requirements.txt"
    if not path.is_file():
        failures.append("requirements.txt is missing")
        return
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not entries:
        failures.append("requirements.txt has no active entries")
    print(f"ok   requirements.txt ({len(entries)} entries)")


def check_expected_paths() -> None:
    for expected in (
        "README.md",
        "LICENSE",
        "launch.bat",
        "requirements.txt",
        "app/__init__.py",
        "app/main.py",
        "app/webui.py",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feedback.yml",
    ):
        if not (ROOT / expected).is_file():
            failures.append(f"missing expected file: {expected}")
    print("ok   expected paths")


def check_version() -> None:
    init = ROOT / "app" / "__init__.py"
    if not init.is_file():
        return
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        init.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        failures.append("app/__init__.py: __version__ not found")
        return
    version = match.group(1)
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        failures.append(f"app/__init__.py: __version__ '{version}' is not semver")
    print(f"ok   __version__ = {version}")


def main() -> int:
    check_yaml()
    check_workflows()
    check_requirements()
    check_expected_paths()
    check_version()

    if failures:
        print("\nFAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
