# pointpillars.c

[![PointPillars live terminal viewer](docs/pointpillars-tui.png)](docs/pointpillars-tui.mp4)

*Click the poster to watch native inference drive a flowing 3D terminal point cloud.*

A small, auditable C11 runtime for the OpenPCDet nuScenes PointPillars
MultiHead checkpoint. It turns ten lidar sweeps into 3D detections without
Python at runtime, and runs natively on macOS, Linux, and WSL.

```text
265,562 points → 28,517 pillars → 69 boxes
Apple M2 · Accelerate/BNNS · 302.983 ms warm median
```

## The idea

PointPillars becomes dense after scatter. The runtime keeps the sparse front
end simple, reuses bounded dense activation arenas, maps a 23 MiB frozen model
directly into memory, and gives each platform an explicit acceleration path:

- Accelerate/BNNS on Apple Silicon;
- OpenMP and AVX2/FMA on x86 CPU;
- custom CUDA or strict FP32 cuDNN on NVIDIA GPUs.

Every fast path has a named correctness boundary. Approximate CUDA is never
presented as graph-equivalent; strict CPU, Apple, precise CUDA, and cuDNN paths
are checked against the original PyTorch checkpoint.

## What is included

- the full 190-tensor PFN, BEV backbone, six prediction heads, velocity decode,
  and rotated per-class NMS;
- deterministic `.pth` → `.ppw` conversion and OpenPCDet-compatible ten-sweep
  nuScenes preparation;
- single-frame, benchmark, batch, and interactive terminal modes;
- reproducible JSON performance reports and official nuScenes evaluation;
- a responsive ANSI/Braille 3D point cloud and BEV with sweep flow, height/age
  shading, filters, trails, selection, camera controls, and terminal recovery.

## Quick start

Put `pp_multihead_nds5823_updated.pth` in `ckpts/`, then:

```sh
make setup-model
make model PYTHON=.venv/bin/python
make
make test PYTHON=.venv/bin/python
```

macOS automatically selects Accelerate/BNNS and does not require Homebrew
OpenMP. Linux/WSL uses the native OpenMP CPU path. CUDA is optional.

Prepare nuScenes mini after extracting it to `/data/nuscenes`:

```sh
make prepare-data PYTHON=.venv/bin/python

frame=$(find /data/nuscenes/pointpillars_10sweep \
  -name '*.bin' -type f | sort | sed -n '2p')

./build/pointpillars infer \
  nuscenes_multihead.ppw "$frame" result.ppout 5 boxes.json

./build/pointpillars tui \
  nuscenes_multihead.ppw /data/nuscenes/pointpillars_10sweep
```

Use `m` to switch between perspective 3D and metric BEV, `f` to freeze sweep
flow, `i` for the inspector, `Space` to pause, arrows to step, `WASD` to pan,
and `z`/`e` to rotate. The [TUI chapter](wiki/08-terminal-visualizer.md) has the
full interaction map.

## Honest numbers

The current macOS review uses a real 265,562-point, ten-sweep nuScenes mini
frame on an 8-core Apple M2. Run zero is cold; the table reports 19 warm runs.

| Path | Warm median | p95 | Raw checkpoint oracle |
|---|---:|---:|---:|
| Apple Accelerate/BNNS | **302.983 ms** | 315.578 ms | max abs `8.96e-4`, pass |
| GGML build on macOS | 305.907 ms | 307.515 ms | same strict output |

GGML is not promoted on macOS because Accelerate already handles its candidate
shapes first. On the full mini set, CPU batch produced 22,109 boxes across
404/404 frames. Official `mini_val` evaluation reproduced mAP `0.2055` and NDS
`0.3280`. The checked WSL/NVIDIA reference reaches 12.160 ms with strict cuDNN
compact detection; fixtures and machines differ, so the numbers are not mixed.

Reproduce the local report and oracle:

```sh
make perf-cpu PERF_FRAME="$frame" PERF_REPS=20 PERF_THREADS=8 \
  PYTHON=.venv/bin/python
make checkpoint-oracle PERF_FRAME="$frame" PYTHON=.venv/bin/python
```

## Documentation

The README is the entry point; implementation and measurement detail lives in
the wiki:

- [system walkthrough](wiki/Home.md)
- [model contract](wiki/01-model-contract.md)
- [CPU and Apple acceleration](wiki/04-cpu-inference.md)
- [CUDA and cuDNN](wiki/05-cuda-inference.md)
- [terminal viewer](wiki/08-terminal-visualizer.md)
- [correctness funnel](wiki/09-validation.md)
- [performance workflow](wiki/11-performance-workflow.md)
- [macOS + real nuScenes mini review](wiki/13-local-macos-nuscenes-mini.md)

## License

See [LICENSE](LICENSE).
