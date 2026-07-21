#!/usr/bin/env python3
"""Check the dependency-free technical article and generated model bundle."""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPECTED_IDS = {
    "flow-rail", "flow-index", "flow-name", "flow-shape", "flow-note",
    "operator-composition", "group-filters", "module-search", "module-list",
    "browser-summary", "perf-tabs", "stage-chart", "performance-ladder",
}


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"] or "")
        attribute = "href" if tag in {"a", "link"} else "src" if tag in {"img", "script"} else None
        if attribute and values.get(attribute):
            self.references.append(values[attribute] or "")


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> int:
    parser = PageParser()
    parser.feed((DOCS / "index.html").read_text(encoding="utf-8"))
    missing_ids = EXPECTED_IDS - parser.ids
    if missing_ids:
        fail(f"index.html is missing interactive IDs: {sorted(missing_ids)}")
    for reference in parser.references:
        if reference.startswith(("https://", "http://", "#", "mailto:")):
            continue
        path = (DOCS / reference.split("#", 1)[0].split("?", 1)[0]).resolve()
        if DOCS.resolve() not in path.parents and path != DOCS.resolve():
            fail(f"local reference escapes docs/: {reference}")
        if not path.exists():
            fail(f"missing local reference: {reference}")

    data = json.loads((DOCS / "model-data.json").read_text(encoding="utf-8"))
    expected = {"schema_version", "model", "config", "summary", "flow", "groups", "modules", "performance"}
    if set(data) != expected or data["schema_version"] != 1:
        fail("unexpected model-data schema")
    checks = {
        "stored tensors": (data["model"]["tensors"], 190),
        "grouped entries": (data["summary"]["modules"], 97),
        "learned operators": (data["summary"]["learned_operators"], 93),
        "flow stages": (len(data["flow"]), 12),
        "model groups": (len(data["groups"]), 6),
        "performance reports": (len(data["performance"]["reports"]), 5),
    }
    for label, (actual, wanted) in checks.items():
        if actual != wanted:
            fail(f"{label}: expected {wanted}, got {actual}")
    if sum(module["elements"] for module in data["modules"]) != data["model"]["elements"]:
        fail("module element total does not match the PPW summary")
    if sum(group["modules"] for group in data["groups"]) != len(data["modules"]):
        fail("group module total does not match the inventory")
    if len({module["name"] for module in data["modules"]}) != len(data["modules"]):
        fail("module names are not unique")
    report_ids = {report["id"] for report in data["performance"]["reports"]}
    if report_ids != {"cpu", "cuda_raw", "cuda_compact", "cudnn_raw", "cudnn_compact"}:
        fail(f"unexpected performance report IDs: {sorted(report_ids)}")
    if data["performance"]["model_sha256"] != data["model"]["sha256"]:
        fail("performance reports and model inventory have different model hashes")
    if len(data["performance"]["point_sha256"]) != 64:
        fail("performance fixture SHA-256 is malformed")
    if not re.search(r"fetch\([\"']model-data\.json[\"']\)", (DOCS / "site.js").read_text()):
        fail("site.js does not load the generated model bundle")
    print("site check: 97 model entries, 12 tensor stages, 5 benchmark reports, all local assets present")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"site check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
