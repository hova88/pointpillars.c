# PointPillars.c: An Inference Engine Seen Through Its Memory Traffic

> **Outcome.** This repository executes a native OpenPCDet nuScenes PointPillars MultiHead checkpoint in dependency-free C, with optional CUDA. The interesting story is not that the model fits in a small binary. It is how the runtime moves a 229,532-point frame through sparse host preprocessing, a 64 MiB BEV canvas, 133.57 GFLOP of convolution, 36 prediction branches, compact device-to-host transfer, rotated NMS, and an interactive terminal renderer.

![End-to-end PointPillars execution pipeline](assets/pipeline.svg)

*The runtime is easiest to understand as a sequence of residency and representation changes.*

## The performance ladder

The measurements below use one 229,532-point, ten-sweep nuScenes frame with 7,868 live pillars on an Intel i5-14600KF and RTX 4060 Ti. CPU uses 16 OpenMP threads. CUDA means the default hybrid FP16-WMMA path; the first CUDA call includes allocation and upload, so the table reports warm means.

| Path | Warm network latency | Effective network rate | Primary mechanism | Correctness |
|---|---:|---:|---|---|
| CPU AVX2/FMA | 591.99 ms | 0.226 TFLOP/s | four output channels reuse each input vector | PyTorch oracle gate |
| CUDA hybrid | 57.50 ms | 2.32 TFLOP/s | implicit-WMMA backbone, explicit reused heads | `allclose=True` |
| CUDA precise | slower by design | — | direct FP32 reductions | max CPU delta below 1e-3 |

The effective rates divide a source-derived `133.57 GFLOP` by measured latency. They are not profiler FLOP counts and they include layout work, activations, synchronization, and transfers.

## Read this like a worklog

1. [The frozen model contract](01-model-contract.md) — shapes, layouts, classes, and why specialization is legitimate.
2. [From ten sweeps to pillars](02-voxelization.md) — sparse indexing, cache lines, and eliminating full-capacity clears.
3. [A 23 MiB mapped model container](03-model-container.md) — offline folding, aligned records, CRCs, and zero-copy ownership.
4. [The CPU path](04-cpu-inference.md) — NCHW loop order, L1-sized row tiles, AVX2/FMA, gathers, and OpenMP ownership.
5. [The CUDA path](05-cuda-inference.md) — tensor-core tiling, explicit versus implicit im2col, events, and residency.
6. [Decode and rotated NMS](06-decode-and-nms.md) — avoiding 590k transcendental calls and retaining one canonical box representation.
7. [The whole pipeline](07-pipeline-and-memory.md) — bounded double buffering, cold versus warm latency, peak memory, and compact transfer.
8. [The terminal as a point-cloud UI](08-terminal-visualizer.md) — Braille pixels, metric layers, tracking, and terminal recovery.
9. [The correctness funnel](09-validation.md) — fixtures, oracle equivalence, official nuScenes evaluation, and what each claim proves.
10. [Extension map](10-extension-guide.md) — where to add a backend, operator, output format, or visualization layer without breaking the contract.

## Repository map

| Area | Role |
|---|---|
| [`src/voxel.c`](../src/voxel.c) | file loading and deterministic pillar features |
| [`src/model.c`](../src/model.c) | memory-mapped, bounds-checked model container |
| [`src/infer_cpu.c`](../src/infer_cpu.c) | complete C11/OpenMP/AVX2 inference backend |
| [`src/infer_cuda.cu`](../src/infer_cuda.cu) | optional persistent CUDA backend |
| [`src/decode.c`](../src/decode.c) | score filtering, residual decode, rotated NMS |
| [`src/main.c`](../src/main.c) | CLI, bounded preprocessing pipeline, serialization |
| [`src/tui.c`](../src/tui.c) | interactive ANSI/Braille visualization |
| [`tools/`](../tools) | offline export, oracle, dataset preparation, evaluation |
| [`tests/`](../tests) | focused C fixtures for the runtime contracts |

## Reproduce the two headline numbers

```sh
make model
make
make cuda

frame=$(find /data/nuscenes/pointpillars_10sweep -name '*.bin' | head -1)
OMP_NUM_THREADS=16 ./build/pointpillars bench nuscenes_multihead.ppw "$frame" 5
./build/pointpillars_cuda bench-cuda nuscenes_multihead.ppw "$frame" 12
```

The commands intentionally keep the first run visible. Warm means exclude it because CUDA allocation, weight upload, FP16 conversion, and page faults are real cold-start work, but they are not paid on every frame.

## What to remember

- The model is dense after scatter, so resident weights and bounded activation arenas beat clever weight streaming.
- CPU and GPU want different representations, and even one GPU graph wants different convolution strategies for backbone and heads.
- The runtime treats correctness as a ladder: container safety, operator fixtures, graph equivalence, decoded equivalence, then task accuracy.

## What remains

The implementation still leaves testable hypotheses: CUDA Graph capture may reduce launch overhead, head shapes may benefit from more specialized WMMA layouts, and the CPU workspace could become persistent across calls. Those are extension points, not claims of free speedup.
