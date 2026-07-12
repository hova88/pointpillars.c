#!/usr/bin/env python3
"""Validate local Markdown links, image alt text, and English-only prose."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

LINK = re.compile(r"(!?)\[([^]]*)\]\(([^)]+)\)")
CJK = re.compile(r"[\u3400-\u9fff]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repository-root", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    allowed_root = (args.repository_root or root.parent).resolve()
    errors: list[str] = []
    pages = sorted(root.rglob("*.md"))
    if not pages:
        errors.append(f"no Markdown pages under {root}")
    for page in pages:
        text = page.read_text(encoding="utf-8")
        rel = page.relative_to(root)
        if CJK.search(text):
            errors.append(f"{rel}: contains CJK text; publication copy must be English")
        if not re.search(r"^# .+", text, re.M):
            errors.append(f"{rel}: missing H1")
        for image, label, target in LINK.findall(text):
            target = target.strip().split("#", 1)[0]
            if image and not label.strip():
                errors.append(f"{rel}: image has empty alt text")
            if not target or re.match(r"(?:https?|mailto):", target):
                continue
            resolved = (page.parent / unquote(target)).resolve()
            if allowed_root not in resolved.parents and resolved != allowed_root:
                errors.append(f"{rel}: link escapes repository root: {target}")
            elif not resolved.exists():
                errors.append(f"{rel}: broken local link: {target}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"wiki check: {len(pages)} Markdown pages, all local links valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
