# macOS and Apple Silicon: Strict Acceleration Without CUDA

> **Outcome.** The repository builds and runs natively on macOS with Apple
> Clang. On an 8-core Apple M2, an oracle-gated Accelerate/BNNS hybrid reduces
> warm raw inference from `7017.255 ms` to `249.490 ms` while matching the
> PyTorch checkpoint at `6.51e-5` maximum absolute error.

## Build from a downloaded checkpoint

Place `pp_multihead_nds5823_updated.pth` in `ckpts/`, then run:

```sh
make setup-model
make model PYTHON=.venv/bin/python
make
make test PYTHON=.venv/bin/python
```

The model step creates a 23.19 MiB, 190-tensor `.ppw` file. Python is not used
after conversion. The macOS build automatically:

- selects `OMP=0`, because Apple Clang does not ship the OpenMP runtime;
- links the system `Accelerate.framework`;
- compiles `src/infer_apple.c` behind `PP_WITH_ACCELERATE`;
- uses `<util.h>` for the PTY test and exposes `SIGWINCH` through Darwin's
  feature set;
- uses `@loader_path` and libc++ for the optional GGML target.

No Homebrew compiler or `libomp` is required for the default path. BNNS manages
its own optimized execution and scheduling.

## Why not translate AVX2 to NEON instruction by instruction?

The x86 microkernels reuse eight horizontal pixels across eight output
channels. A literal NEON port would provide only part of the solution: it would
still need packing, shape-specific blocking, core scheduling, and transposed
convolution. BNNS already implements those choices for Apple hardware and
accepts the runtime's existing CHW images and OIHW weights.

The adapter therefore creates fixed-shape filters once, caches them by weight
pointer and convolution geometry, and applies them directly to the existing
activation arenas. There is no per-frame image repack. Two transposed deblocks
need one cold weight transformation because the checkpoint stores
`[Cin,Cout,K,K]` while BNNS exposes an `IOHrWr` rotated view.

## The strict shape gate

An operator fixture is necessary but not sufficient. Small BNNS 3×3 fixtures
reach at most `3.58e-7` error, and the transposed-convolution fixture is exact.
The complete all-BNNS graph is faster, but the checkpoint oracle found:

```text
all-BNNS: max_abs 2.120398, mean_abs 0.001181, allclose False
```

Shape isolation identified the only failing promotion as deblock 0's
`2×2/stride-2` forward convolution. Keeping it on canonical C produces:

```text
strict hybrid: max_abs 6.5088e-5, mean_abs 9.7784e-7, allclose True
```

This is the default. `PP_APPLE_CONV2=1` reproduces the rejected experiment but
is not a supported strict mode. `PP_APPLE_DISABLE=1` restores the complete C
reference, and `PP_APPLE_DECONV_DISABLE=1` isolates the deblock acceleration.

## Measured M2 result

The controlled fixture contains 24,000 points in 7,881 live pillars. Reports
use the raw 3,866,624-float output boundary.

| Path | Cold | Warm median | p95 | Visible workspace |
|---|---:|---:|---:|---:|
| portable C | 7022.236 ms | 7017.255 ms | 7029.027 ms | 129.9 MiB |
| strict Accelerate hybrid | 266.440 ms | 249.490 ms | 253.699 ms | 130.5 MiB |

The strict speedup is `28.1×`. The visible workspace increase is the cached
rotated deblock weights and filter records; BNNS' internal opaque capacity is
not guessed. A semantic detection comparison found the same 252 boxes and
class sequence. Maximum decoded field differences were `8.64e-7` for score,
`4.77e-6` for dimensions, and `3.20e-5` for yaw.

These values are evidence for this synthetic regression fixture, not a nuScenes
throughput or task-accuracy claim. Use prepared real frames and the official
evaluation workflow before publishing dataset results.

## Real nuScenes mini follow-up

The complete local audit also measures a real 265,562-point ten-sweep frame
with 28,517 live pillars. On the same 8-core Apple M2, 19 warm calls have a
`302.983 ms` median and `315.578 ms` p95. The checkpoint oracle passes at
`8.96e-4` maximum and `9.89e-6` mean absolute error. CPU batch completes all
404 mini keyframes, and official `mini_val` evaluation reproduces mAP `0.2055`
and NDS `0.3280`.

See [the complete local run](13-local-macos-nuscenes-mini.md) for archive
placement on macOS, data invariants, command coverage, artifact hashes, and the
real-frame TUI recording.

## Benchmark and validate your Mac

```sh
frame=/path/to/prepared-frame.bin

make perf-cpu PERF_FRAME="$frame" PERF_REPS=20 PERF_THREADS=8 \
  PYTHON=.venv/bin/python

make checkpoint-oracle PERF_FRAME="$frame" \
  PYTHON=.venv/bin/python

PP_APPLE_DISABLE=1 ./build/pointpillars bench \
  nuscenes_multihead.ppw "$frame" 3
```

Thermal state matters on fanless and mobile Macs. Keep the first call visible,
use several warm runs, record `hw.model`/CPU identity, and do not compare a
synthetic fixture with the WSL nuScenes report.

## Portability boundary

All Apple-specific code is isolated behind `PP_WITH_ACCELERATE`. Linux and WSL
continue to compile the existing OpenMP/AVX2 code and do not link Apple
frameworks. The stable public executable name remains `build/pointpillars`, so
scripts and the JSON performance protocol work on both platforms.
