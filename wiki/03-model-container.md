# A 23 MiB Model That Never Gets Copied on CPU

> **Outcome.** The `.ppw` file is a small, versioned contract between an offline PyTorch exporter and the C runtime. CPU inference reads weights directly from a private read-only mapping. CUDA uploads the same data once, derives a persistent FP16 copy once, and reuses both for every frame.

## File layout

[`src/model.c`](../src/model.c) expects a 64-byte header followed by 190 fixed-size tensor records and an aligned data region:

```text
┌──────────────────── 64 B header ────────────────────┐
│ magic · version · count · record size · alignment   │
│ data offset · data bytes · tensor-table CRC32       │
├────────────── 190 × 112 B tensor records ───────────┤
│ name · rank · dims[4] · offset · bytes · CRC32      │
├─────────────── 64-byte aligned fp32 data ───────────┤
│ PFN · backbone · deblocks · shared · 36 head pairs  │
└──────────────────────────────────────────────────────┘
```

The current mapped file is 23.19 MiB. A `pp_model` owns the `mmap` address, byte size, record array, data base, and tensor count. Closing it is one `munmap`; no individual tensor allocation exists.

## Validation before trust

`pp_model_open` checks:

- eight-byte magic and version `2`;
- record count and exact record size;
- 64-byte alignment declaration;
- table and data ranges with overflow-resistant subtraction checks;
- table CRC32;
- NUL-terminated names, rank at most four, and in-range data slices;
- every tensor's payload CRC32.

The loader closes the file descriptor immediately after `mmap`. The mapping, not the descriptor, owns access from then on. `MAP_PRIVATE` plus `PROT_READ` prevents runtime mutation and lets the OS page cache share clean pages.

> **Sidenote — CRC is not authentication.** CRC32 catches truncation and accidental corruption. It does not make an untrusted model file cryptographically safe or authentic.

## Lookup cost and why it is acceptable here

`pp_model_tensor` linearly scans 190 records, compares names, validates rank and dimensions, then returns a pointer into the mapped data. Linear search would be a poor design for a graph with tens of thousands of tensors. Here, the graph is fixed and lookup occurs once per layer launch, while each convolution performs millions or billions of MACs.

On CUDA, the analogous helper resolves the device pointer by adding the record's byte offset to the device weight base. The runtime also creates a half-precision mirror preserving `offset / sizeof(float)`, so a tensor name resolves to the same logical record in either representation.

If tensor lookup ever appears in a profile, the next step is not a general graph engine. It is an exporter-emitted enum or prebound table for these 190 known names.

## Offline fusion changes the memory equation

[`tools/export_checkpoint.py`](../tools/export_checkpoint.py) folds BatchNorm:

```text
scale = gamma / sqrt(variance + epsilon)
W'    = W × scale
b'    = beta - mean × scale
```

This removes BatchNorm reads and writes from the runtime, eliminates running-stat tensors from the hot graph, and lets every convolution fuse ReLU immediately afterward. The epsilon difference between backbone/PFN (`1e-3`) and head layers (`1e-5`) is encoded during export; getting it wrong changes numerical output even when tensor shapes look perfect.

Weights are made contiguous little-endian fp32 before writing. That matters for both CPU vector loads and the one-time CUDA `f2h` conversion. Runtime code never pays for framework strides or transposes.

## Residency by backend

| Backend | fp32 weights | fp16 weights | Per-frame weight traffic across host/device |
|---|---:|---:|---:|
| CPU | 23.19 MiB mapped | none | none |
| CUDA precise | 23.19 MiB device | mirror allocated but unused by direct conv | none after initialization |
| CUDA fast | 23.19 MiB device | 11.59 MiB device | none after initialization |

The first CUDA call is intentionally expensive: it allocates the context, uploads all weights, converts the data region to FP16, and faults workspaces into residency. Warm benchmarks exclude that call but report it separately.

## Tests and failure modes

[`tests/test_model.c`](../tests/test_model.c) checks the 190-tensor count and representative PFN/head shapes. The broader test workflow also feeds truncated model data to the loader and expects rejection. Shape checks occur again at each named layer boundary, so a valid container with the wrong tensor contract fails loudly.

## What to remember

- A specialized runtime file should encode the post-export execution contract, not the training framework's object graph.
- Memory mapping is both an ownership simplification and a zero-copy CPU weight path.
- Bounds, shape, version, and checksum validation belong before inference, where failure is cheap and explainable.

## What remains

The container does not yet carry a full graph identity or config hash. Adding an exporter-generated contract fingerprint would make mismatched YAML/checkpoint pairs easier to diagnose without turning the runtime into a config parser.
