#!/usr/bin/env python3
"""Build the dependency-free PointPillars technical-site data bundle.

The PPW container is authoritative for stored tensor facts.  Activation shapes
come from the frozen C runtime contract.  Benchmark values are copied only from
versioned ``pointpillars-perf-v1`` reports produced by ``tools/perf.py``.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any


MAGIC = b"PPWGT\0\0\0"
HEADER = struct.Struct("<8sIIIIQQI20x")
ENTRY = struct.Struct("<48sI4I4xQQI20x")
OPENPCDET_COMMIT = "233f849829b6ac19afb8af8837a0246890908755"
OPENPCDET_CONFIG = (
    "https://github.com/open-mmlab/OpenPCDet/blob/"
    f"{OPENPCDET_COMMIT}/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml"
)
CONFIG_MARKERS = (
    "NAME: PointPillar", "NAME: PillarVFE", "NUM_FILTERS: [64]",
    "NAME: PointPillarScatter", "NAME: BaseBEVBackbone",
    "LAYER_NUMS: [3, 5, 5]", "NAME: AnchorHeadMulti",
    "SEPARATE_MULTIHEAD: True", "NMS_POST_MAXSIZE: 83",
)

FLOW = (
    ("points", "Ten-sweep points", "float32 [N, 5]", "x, y, z, intensity, time lag"),
    ("voxel", "Pillarize", "[P, 20, 11] + coords [P, 4]", "P ≤ 30,000; 0.2 m XY cells"),
    ("pfn", "PFN", "[P, 64]", "folded linear + ReLU + max over 20 points"),
    ("bev", "Sparse BEV boundary", "CPU [64, P] · CUDA [1, 64, 512, 512]", "implicit zeros or 64 MiB scatter"),
    ("backbone0", "Backbone 0", "[1, 64, 256, 256]", "stride 2 + three 3×3 convolutions"),
    ("backbone1", "Backbone 1", "[1, 128, 128, 128]", "stride 2 + five 3×3 convolutions"),
    ("backbone2", "Backbone 2", "[1, 256, 64, 64]", "stride 2 + five 3×3 convolutions"),
    ("deblocks", "Three deblocks", "3 × [1, 128, 128, 128]", "downsample · identity · upsample"),
    ("concat", "Aliased concat", "[1, 384, 128, 128]", "three destinations already occupy adjacent slices"),
    ("shared", "Shared convolution", "[1, 64, 128, 128]", "common input for 36 branch middles"),
    ("heads", "Six multi-head groups", "[1, 236, 128, 128]", "class + 10 regression codes per anchor"),
    ("decode", "Decode + rotated NMS", "≤ 830 pp_box values", "109 detections on the reference frame"),
)

GROUPS = (
    ("pfn", "Pillar feature network", "pfn.", 0.0, "11 features become one 64-channel vector per live pillar"),
    ("backbone", "BEV backbone", "backbone.", 36.239, "Sixteen dense 3×3 convolutions across three scales"),
    ("deblocks", "Deblocks", "deblock.", 2.953, "Normalize all scales to 128×128 and 128 channels"),
    ("shared", "Shared feature", "shared.", 3.624, "One 384→64 convolution shared by every prediction branch"),
    ("heads", "Multi-head predictions", "head.", 23.970, "Six groups × six branches × middle/output convolution"),
    ("metadata", "Frozen metadata", "meta.", 0.0, "Range, voxel geometry, anchors and class labels"),
)

REPORT_NAMES = {
    "cpu": "CPU · OpenMP/AVX2 raw",
    "cuda_raw": "Custom CUDA · raw",
    "cuda_compact": "Custom CUDA · compact",
    "cudnn_raw": "cuDNN FMA · raw",
    "cudnn_compact": "cuDNN FMA · compact",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_for(name: str) -> str:
    for group_id, _title, prefix, _gmac, _description in GROUPS:
        if name.startswith(prefix):
            return group_id
    raise ValueError(f"unknown PPW tensor prefix: {name}")


def module_name(name: str) -> str:
    for suffix in (".weight", ".bias"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def operator_for(name: str) -> str:
    if name == "pfn":
        return "PFN affine + max"
    if name.startswith("meta."):
        return "Frozen metadata"
    if name == "deblock.2":
        return "ConvTranspose2d"
    return "Conv2d"


def read_model(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    blob = path.read_bytes()
    if len(blob) < HEADER.size:
        raise ValueError("PPW model is truncated")
    magic, version, count, record_size, alignment, data_offset, data_bytes, table_crc = HEADER.unpack_from(blob)
    table_end = HEADER.size + count * record_size
    if (magic != MAGIC or version != 2 or count > 256 or record_size != ENTRY.size or
            alignment != 64 or data_offset < table_end or data_offset + data_bytes > len(blob)):
        raise ValueError("invalid PPW v2 header")
    table = blob[HEADER.size:table_end]
    if binascii.crc32(table) & 0xFFFFFFFF != table_crc:
        raise ValueError("PPW tensor-table CRC32 mismatch")
    tensors = []
    for index in range(count):
        fields = ENTRY.unpack_from(blob, HEADER.size + index * ENTRY.size)
        raw_name, rank, *tail = fields
        dims, offset, size, crc = tail[:4], tail[4], tail[5], tail[6]
        name = raw_name.split(b"\0", 1)[0].decode("utf-8")
        if rank > 4 or offset + size > data_bytes:
            raise ValueError(f"invalid tensor record: {name}")
        payload = blob[data_offset + offset:data_offset + offset + size]
        if binascii.crc32(payload) & 0xFFFFFFFF != crc:
            raise ValueError(f"tensor CRC32 mismatch: {name}")
        shape = list(dims[:rank])
        tensors.append({"name": name, "shape": shape, "elements": size // 4,
                        "bytes": size, "group": group_for(name)})
    model = {"file": path.name, "sha256": sha256(path), "file_bytes": len(blob),
             "payload_bytes": sum(item["bytes"] for item in tensors),
             "tensors": len(tensors), "elements": sum(item["elements"] for item in tensors)}
    return model, tensors


def build_modules(tensors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for tensor in tensors:
        grouped.setdefault(module_name(tensor["name"]), []).append(tensor)
    group_order = {item[0]: index for index, item in enumerate(GROUPS)}
    modules = []
    for name, values in grouped.items():
        modules.append({"name": name, "group": values[0]["group"],
                        "operator": operator_for(name),
                        "tensor_names": [value["name"] for value in values],
                        "tensor_shapes": [value["shape"] for value in values],
                        "elements": sum(value["elements"] for value in values),
                        "bytes": sum(value["bytes"] for value in values)})
    modules.sort(key=lambda item: group_order[item["group"]])
    return modules


def compact_report(report_id: str, path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "pointpillars-perf-v1":
        raise ValueError(f"{path}: expected pointpillars-perf-v1")
    stages = {name: report["warm"][name]["median"]
              for name in ("pfn", "scatter", "backbone", "heads")}
    return {"id": report_id, "name": REPORT_NAMES[report_id],
            "timestamp_utc": report["timestamp_utc"], "backend": report["backend"],
            "output_mode": report["protocol"]["output_mode"],
            "repetitions": report["protocol"]["repetitions"],
            "warmup_runs": report["protocol"]["warmup_runs"],
            "cold_ms": report["cold"][0]["total"],
            "warm_median_ms": report["warm"]["total"]["median"],
            "warm_p95_ms": report["warm"]["total"]["p95"],
            "stages_ms": stages, "workspace_bytes": report["workspace_bytes"],
            "device_to_host_bytes": report["device_to_host_bytes"],
            "binary_sha256": report["artifacts"]["binary"]["sha256"]}


def validate_config(path: Path) -> dict[str, str]:
    content = path.read_text(encoding="utf-8")
    missing = [marker for marker in CONFIG_MARKERS if marker not in content]
    if missing:
        raise ValueError(f"{path}: frozen PointPillars markers missing: {', '.join(missing)}")
    return {"file": path.as_posix(), "sha256": sha256(path),
            "openpcdet_commit": OPENPCDET_COMMIT,
            "openpcdet_config": OPENPCDET_CONFIG}


def render_markdown(data: dict[str, Any]) -> str:
    model, summary = data["model"], data["summary"]
    lines = [
        "# PointPillars frozen-model summary", "",
        "> Generated by `tools/build_site_data.py`; do not edit by hand.", "",
        "PPW records are authoritative for stored tensor names and shapes. Activation shapes",
        "come from the specialized C runtime; timings come from `pointpillars-perf-v1` reports.", "",
        f"- PPW SHA-256: `{model['sha256']}`",
        f"- Stored tensors: **{model['tensors']}**",
        f"- Grouped entries: **{summary['modules']}**",
        f"- Learned operators: **{summary['learned_operators']}**",
        f"- Stored elements: **{model['elements']:,}**",
        f"- Model file: **{model['file_bytes'] / 2**20:.2f} MiB**",
        f"- Dense convolution work: **{summary['convolution_gmac']:.3f} GMAC**", "",
        f"Architecture reference: [OpenPCDet `cbgs_pp_multihead.yaml`]({data['config']['openpcdet_config']})",
        "at the pinned commit recorded in `model-data.json`.", "",
        "## Activation tensor pipe", "",
        "| # | Operator | Output contract | Note |", "|---:|---|---|---|",
    ]
    for index, item in enumerate(data["flow"], 1):
        lines.append(f"| {index} | {item['operator']} | `{item['shape']}` | {item['note']} |")
    lines += ["", "## Operator composition", "",
              "| Group | Entries | Tensors | GMAC | Operator mix |",
              "|---|---:|---:|---:|---|"]
    for group in data["groups"]:
        mix = ", ".join(f"{name} × {count}" for name, count in group["operators"].items())
        lines.append(f"| {group['title']} | {group['modules']} | {group['tensors']} | {group['gmac']:.3f} | {mix} |")
    if data["performance"]["reports"]:
        lines += ["", "## Measured performance", "",
                  "One cold call is kept visible; medians and p95 values use the remaining warm calls.", "",
                  "| Path | Cold | Warm median | p95 | Workspace | D2H |",
                  "|---|---:|---:|---:|---:|---:|"]
        for report in data["performance"]["reports"]:
            lines.append(f"| {report['name']} | {report['cold_ms']:.3f} ms | {report['warm_median_ms']:.3f} ms | {report['warm_p95_ms']:.3f} ms | {report['workspace_bytes'] / 2**20:.2f} MiB | {report['device_to_host_bytes'] / 1024:.2f} KiB |")
    lines += ["", "## Stored operator inventory", "",
              "| # | Group | Module | Operator | Tensor shape(s) | Elements |",
              "|---:|---|---|---|---|---:|"]
    titles = {group["id"]: group["title"] for group in data["groups"]}
    for index, module in enumerate(data["modules"], 1):
        shapes = " · ".join("[" + ", ".join(map(str, shape)) + "]" for shape in module["tensor_shapes"])
        lines.append(f"| {index} | {titles[module['group']]} | `{module['name']}` | {module['operator']} | `{shapes}` | {module['elements']:,} |")
    lines += ["", "## Evidence boundaries", "",
              "- BatchNorm is folded offline, so it is not a runtime operator.",
              "- Voxelization, sparse lookup/scatter, concatenation, activation, decode, and rotated NMS own no learned tensor row.",
              "- Custom FP16 WMMA timings are approximate; the slower `PP_CUDA_PRECISE=1` route is the graph-equivalence oracle.",
              "- cuDNN timings use deterministic FP32/FMA; TF32 remains an opt-in approximation.",
              "- These are single-fixture latency and graph-equivalence results, not nuScenes mAP/NDS claims.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("nuscenes_multihead.ppw"))
    parser.add_argument("--config", type=Path, default=Path("cfgs/pointpillars.yaml"))
    parser.add_argument("--report", action="append", default=[], metavar="ID=PATH")
    parser.add_argument("--output", type=Path, default=Path("docs/model-data.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/model-summary.md"))
    args = parser.parse_args()
    try:
        model, tensors = read_model(args.model)
        modules = build_modules(tensors)
        reports = []
        for value in args.report:
            report_id, separator, raw_path = value.partition("=")
            if not separator or report_id not in REPORT_NAMES:
                raise ValueError(f"invalid --report {value!r}; expected one of {', '.join(REPORT_NAMES)}")
            reports.append(compact_report(report_id, Path(raw_path)))
        if reports:
            fixture_reports = [json.loads(Path(value.partition("=")[2]).read_text()) for value in args.report]
            fixture = fixture_reports[0]["fixture"]
            point_hash = fixture_reports[0]["artifacts"]["points"]["sha256"]
            model_hash = fixture_reports[0]["artifacts"]["model"]["sha256"]
            if any(item["fixture"] != fixture or item["artifacts"]["points"]["sha256"] != point_hash
                   or item["artifacts"]["model"]["sha256"] != model_hash for item in fixture_reports):
                raise ValueError("performance reports do not share one model and fixture")
            performance = {"fixture": fixture, "point_sha256": point_hash,
                           "model_sha256": model_hash,
                           "machine": fixture_reports[0]["machine"], "reports": reports}
        else:
            performance = {"fixture": None, "reports": []}
        groups = []
        for group_id, title, prefix, gmac, description in GROUPS:
            selected = [module for module in modules if module["group"] == group_id]
            groups.append({"id": group_id, "title": title, "prefix": prefix,
                           "description": description, "gmac": gmac,
                           "modules": len(selected),
                           "tensors": sum(len(module["tensor_names"]) for module in selected),
                           "elements": sum(module["elements"] for module in selected),
                           "operators": dict(sorted(Counter(module["operator"] for module in selected).items()))})
        output = {"schema_version": 1, "model": model,
                  "config": validate_config(args.config),
                  "summary": {"modules": len(modules), "learned_operators": 93,
                              "convolution_gmac": 66.786},
                  "flow": [dict(zip(("id", "operator", "shape", "note"), row)) for row in FLOW],
                  "groups": groups, "modules": modules, "performance": performance}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(output), encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"build_site_data: {exc}\n")
    print(f"wrote {args.output} and {args.markdown}: {model['tensors']} tensors, {len(modules)} grouped entries, "
          f"{len(reports)} performance reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
