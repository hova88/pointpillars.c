#!/usr/bin/env python3
"""Small dependency-free tests for the PPW site-data parser."""

from __future__ import annotations

import binascii
import importlib.util
import struct
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_site_data", ROOT / "tools" / "build_site_data.py")
assert SPEC and SPEC.loader
site_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(site_data)


def make_ppw(path: Path) -> None:
    records = [
        ("pfn.weight", (2, 3), struct.pack("<6f", *range(6))),
        ("pfn.bias", (2,), struct.pack("<2f", 1.0, 2.0)),
        ("meta.range", (6,), struct.pack("<6f", *range(6))),
    ]
    table_size = len(records) * site_data.ENTRY.size
    data_offset = site_data.HEADER.size + table_size
    entries = []
    payload = bytearray()
    cursor = 0
    for name, shape, raw in records:
        dims = tuple(shape) + (0,) * (4 - len(shape))
        entries.append(site_data.ENTRY.pack(
            name.encode().ljust(48, b"\0"), len(shape), *dims, cursor, len(raw),
            binascii.crc32(raw) & 0xFFFFFFFF))
        payload.extend(raw)
        cursor += len(raw)
    table = b"".join(entries)
    header = site_data.HEADER.pack(site_data.MAGIC, 2, len(records), site_data.ENTRY.size, 64,
                                   data_offset, len(payload),
                                   binascii.crc32(table) & 0xFFFFFFFF)
    path.write_bytes(header + table + payload)


def test_parser_and_grouping(tmp: Path) -> None:
    model_path = tmp / "tiny.ppw"
    make_ppw(model_path)
    model, tensors = site_data.read_model(model_path)
    assert model["tensors"] == 3
    assert model["elements"] == 14
    modules = site_data.build_modules(tensors)
    assert [item["name"] for item in modules] == ["pfn", "meta.range"]
    assert modules[0]["operator"] == "PFN affine + max"
    assert modules[0]["tensor_shapes"] == [[2, 3], [2]]
    assert site_data.group_for("head.5.vel.conv2.weight") == "heads"
    assert site_data.module_name("backbone.block1.conv3.bias") == "backbone.block1.conv3"


def test_crc_rejection(tmp: Path) -> None:
    model_path = tmp / "corrupt.ppw"
    make_ppw(model_path)
    raw = bytearray(model_path.read_bytes())
    raw[-1] ^= 0xFF
    model_path.write_bytes(raw)
    try:
        site_data.read_model(model_path)
    except ValueError as error:
        assert "CRC" in str(error)
    else:
        raise AssertionError("corrupt payload was accepted")


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        test_parser_and_grouping(tmp)
        test_crc_rejection(tmp)
    print("site data tests: parser, grouping, and CRC rejection passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
