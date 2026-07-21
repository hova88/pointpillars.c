# pointpillars.c

A small, auditable C11 runtime for the OpenPCDet nuScenes PointPillars
MultiHead checkpoint. It turns ten lidar sweeps into 3D detections without
Python at runtime, and runs natively on macOS, Linux, and WSL.

**[Read the visual technical article →](https://hova88.github.io/pointpillars.c/)**

The article walks through every tensor boundary, the 93 learned operators,
memory residency, correctness levels, and five measured CPU/CUDA/cuDNN routes.
Its interactive model explorer is also available as a generated
[plain-Markdown summary](docs/model-summary.md).

[![PointPillars terminal viewer](docs/pointpillars-tui.gif)](docs/pointpillars-tui.mp4)

*The GIF plays directly on GitHub; click it for the original MP4.*

```text
26,414 points → 7,854 pillars → 109 boxes
Intel i5-14600KF · OpenMP/AVX2 · 503.105 ms warm median
```

## The idea

PointPillars becomes dense at the first BEV convolution. The CPU route reads
live pillars directly through the voxel grid, while accelerator routes may
materialize scatter. The runtime reuses bounded activation arenas, maps a
23 MiB frozen model directly into memory, and gives each platform an explicit
acceleration path:

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
- reproducible JSON performance reports and checkpoint-oracle validation;
- a responsive ANSI/Braille 3D point cloud and BEV where every in-view point
  contributes through density-aware rendering, with height/age shading,
  filters, trails, selection, camera controls, and terminal recovery.

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

Use `m` to switch between perspective 3D and metric BEV, `i` for the inspector,
`Space` to pause, arrows to step, `WASD` to pan,
and `z`/`e` to rotate. The [TUI chapter](wiki/08-terminal-visualizer.md) has the
full interaction map.

## Honest numbers

The current CPU review uses one real 26,414-point frame with 7,854 live
pillars on a 16-thread Intel i5-14600KF under WSL. Three alternating report
pairs contribute 36 warm samples per route in the same binary.

| First BEV route | Warm median | Visible workspace |
|---|---:|---:|
| direct sparse grid | **503.105 ms** | **65.917 MiB** |
| materialized dense (`PP_CPU_DENSE_FIRST=1`) | 518.493 ms | 129.917 MiB |

The sparse route is 2.97% faster by median and removes the 64 MiB scatter
arena. Its PyTorch checkpoint comparison passes with max absolute error
`0.0010023` and mean absolute error `1.06e-5`.

Reproduce the local report and oracle:

```sh
make perf-cpu PERF_FRAME="$frame" PERF_REPS=20 PERF_THREADS=16
make checkpoint-oracle PERF_FRAME="$frame"
```

## Documentation

The README is the entry point. The
[visual systems walkthrough](https://hova88.github.io/pointpillars.c/) teaches
the pipeline in one page; implementation and measurement detail lives in the
wiki:

- [system walkthrough](wiki/Home.md)
- [model contract](wiki/01-model-contract.md)
- [CPU and Apple acceleration](wiki/04-cpu-inference.md)
- [CUDA and cuDNN](wiki/05-cuda-inference.md)
- [terminal viewer](wiki/08-terminal-visualizer.md)
- [correctness funnel](wiki/09-validation.md)
- [performance workflow](wiki/11-performance-workflow.md)
- [macOS + real nuScenes mini review](wiki/13-local-macos-nuscenes-mini.md)

The article is plain HTML/CSS/JavaScript under `docs/`. Validate it without a
checkpoint or Python packages:

```sh
make site-check
python3 -m http.server 8000 --directory docs
```

See [docs/PUBLISHING.md](docs/PUBLISHING.md) to regenerate the checked model
inventory and benchmark bundle.

## License

See [LICENSE](LICENSE).
