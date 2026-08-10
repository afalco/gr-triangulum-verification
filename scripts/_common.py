"""Shared helpers for the verification scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Report:
    """Minimal pass/fail reporter with a non-zero exit code on failure."""

    def __init__(self, title: str):
        self.title = title
        self.failures: list[str] = []
        self.warnings: list[str] = []
        print("=" * 78)
        print(title)
        print("=" * 78)

    def section(self, text: str) -> None:
        print(f"\n--- {text} " + "-" * max(0, 72 - len(text)))

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)
        return ok

    def warn(self, label: str, detail: str = "") -> None:
        print(f"  [WARN] {label}" + (f"  {detail}" if detail else ""))
        self.warnings.append(label)

    def info(self, text: str) -> None:
        print(f"         {text}")

    def finish(self, exit_on_failure: bool = True) -> int:
        print("\n" + "=" * 78)
        if self.failures:
            print(f"RESULT: {len(self.failures)} FAILED, "
                  f"{len(self.warnings)} warning(s)")
            for f in self.failures:
                print(f"  - {f}")
        else:
            print(f"RESULT: all checks passed, "
                  f"{len(self.warnings)} warning(s)")
        print("=" * 78)
        code = 1 if self.failures else 0
        if exit_on_failure:
            sys.exit(code)
        return code
