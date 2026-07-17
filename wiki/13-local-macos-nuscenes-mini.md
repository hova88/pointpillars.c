# A Complete Local Run on macOS and nuScenes Mini

> **Outcome.** The July 2026 local audit exercised every backend and workflow
> that is meaningful on an 8-core Apple M2: model inspection, unit and PTY
> tests, real ten-sweep preparation, single-frame inference, warm benchmarks,
> structured performance capture, PyTorch oracle comparison, all-frame batch,
> official `mini_val` evaluation, GGML comparison, interactive TUI capture,
> and H.264 decode verification.

## Put nuScenes at `/data/nuscenes` on macOS

macOS keeps `/` on a sealed read-only APFS volume. The writable backing store
for a conventional `/data` root is `/System/Volumes/Data/data`; a synthetic
link makes it visible at `/data` after boot.

```sh
mkdir -p /System/Volumes/Data/data/nuscenes
tar -xf ~/Downloads/Nuscenes-v1.0-mini.tar \
  -C /System/Volumes/Data/data/nuscenes

# One-time administrator step. The tab between the two fields is required.
sudo mkdir -p /etc/synthetic.d
printf 'data\tSystem/Volumes/Data/data\n' | \
  sudo tee /etc/synthetic.d/pointpillars-data
sudo reboot
```

After the reboot, `/data/nuscenes` and
`/System/Volumes/Data/data/nuscenes` refer to the same files. Using a dedicated
manifest in `/etc/synthetic.d` avoids overwriting other system mappings.

The archive was inspected for absolute paths and `..` traversal before
extraction. The extracted mini set contains:

| Item | Count / size |
|---|---:|
| total files | 31,224 |
| extracted size | 5.1 GiB |
| lidar keyframes | 404 |
| lidar sweeps | 3,531 |
| metadata version | `v1.0-mini` |

All metadata tables parsed as JSON before preparation.

## Prepare the real model input

```sh
make prepare-data PYTHON=.venv/bin/python
```

The converter follows each keyframe's lidar history, transforms up to ten
sweeps into the keyframe sensor frame, removes ego returns, normalizes
intensity, writes time lag into feature five, and deterministically shuffles
the result.

| Prepared invariant | Result |
|---|---:|
| files and manifest rows | 404 / 404 |
| total points | 102,505,809 |
| points per frame | 20,504–279,217 |
| frames with all ten sweeps | 394 |
| binary size check | every file equals `points × 5 × sizeof(float)` |

The first frame of each scene legitimately has fewer historical sweeps. The
representative benchmark therefore uses the second sorted frame:

```text
1532402928147847_39586f9d59004284a7114a68825e8eec.bin
SHA-256 89228db11d064b0f193600e215c7195606fbbed2abb6f7a6d65c3d50824bef36
265,562 points · 169,690 accepted · 28,517 pillars · 23,055 clipped
```

## Local execution matrix

| Workflow | Evidence | Result |
|---|---|---|
| build + unit tests | `make test` | model, voxel, decode, convolution, Apple deconv, PTY/TUI pass |
| model inspection | `pointpillars inspect` | 190 tensors, 23.19 MiB mapped |
| single frame | `pointpillars infer` | 69 decoded boxes and raw `.ppout` |
| native warm report | `make perf-cpu` | 20 runs captured in `build/perf/cpu.json` |
| raw graph oracle | `make checkpoint-oracle` | 3,866,624 floats, `allclose=True` |
| full CPU batch | `pointpillars batch` | 404/404 JSON files, no failed frame |
| official evaluation | `make evaluate` | 81 `mini_val` samples evaluated |
| optional GGML build | `make ggml`, `make perf-ggml` | builds and remains strict; no Mac speedup |
| live terminal | `pointpillars tui` through real PTY recorder | 96 interactive frames captured |
| media integrity | `ffprobe` + full `ffmpeg` decode | valid H.264/YUV420P, zero decode errors |

CUDA and cuDNN targets are intentionally not runnable on macOS. They remain
covered by the WSL/NVIDIA reports rather than being represented as local Mac
tests.

## Real Apple M2 performance and correctness

The structured report identifies the binary, model, fixture, git revision,
machine, environment, cold call, and every warm call. The native binary SHA is
`86471b5aba241d7f50bf82329e783ffb848a0602520c639368b8524c5a739520`.

| Path | Cold | Warm median | p95 | Visible workspace |
|---|---:|---:|---:|---:|
| Accelerate/BNNS strict hybrid | 345.392 ms | **302.983 ms** | 315.578 ms | 135.5 MiB |
| GGML build on the same Mac | 328.751 ms | 305.907 ms | 307.515 ms | 135.5 MiB |

The independent checkpoint comparison reports maximum absolute error
`8.9645e-4`, mean absolute error `9.8870e-6`, and `allclose=True`. The real
frame has far more live pillars than the smaller synthetic regression fixture,
so its PFN time and visible workspace are larger.

GGML is not a second accelerator on this Mac build. `pp_apple_conv` is tried
first and accepts the shapes GGML would otherwise handle. The GGML binary is a
useful build/compatibility check, but its measured median is slightly slower;
the default remains the smaller dependency-free Accelerate binary.

## Full mini batch and official metric

The bounded two-slot CPU batch processed every prepared keyframe and wrote 404
JSON files containing 22,109 boxes. Per-frame detections ranged from 15 to 119
with a median of 55. Preparation was normally 5–9 ms, decode normally stayed
below 0.6 ms, and inference varied with scene/pillar count and machine state.

The submission tool selected the two official `mini_val` scenes:

```text
81 samples · 5,345 submitted boxes · 4,890 after official filtering
mAP 0.2055 · NDS 0.3280
mATE 0.5288 · mASE 0.5318 · mAOE 0.6158 · mAVE 0.5699 · mAAE 0.5013
```

This reproduces the checked metrics under `evaluation/nuscenes-mini/`. It does
not claim that the checkpoint filename's `5823` was measured on mini; that
number comes from another/full evaluation context.

The official devkit is a documentation/evaluation dependency, not a C runtime
dependency. Its 1.2.0 release currently pins NumPy below 2, so use Python
3.10–3.13 for the supported environment:

```sh
python3.12 -m venv .venv-eval
.venv-eval/bin/pip install -r requirements-eval.txt
make evaluate EVAL_PYTHON=.venv-eval/bin/python
```

## The checked TUI recording

The README media was regenerated from the native Accelerate executable and
the 404 real prepared frames, not from a synthetic fixture or parallel mock.
The PTY script exercised selection, zoom, rotation, pause, frame stepping,
filters, help, and trails.

```text
120 × 40 terminal · 96 captured frames · 12 FPS · 8.0 seconds
1200 × 720 H.264 · YUV420P · fast-start · 2.68 MiB
MP4 SHA-256 3752af33bd02c8d450fd6489354aaea86c844e95f25f5ade84d609b69f5a7ae3
poster SHA-256 d58a7ee7a7b6905bfc19f12c244480e7bcab34e0b056ac55060e2df44e78e544
```

The selected poster frame shows frame 36/404, 267,752 input points, 102
decoded objects, 29 visible at score 0.20, an active local track, 10 m rings,
and a measured 292.39 ms inference call.

## Reproduce only what you need

```sh
frame=$(find /data/nuscenes/pointpillars_10sweep \
  -name '*.bin' -type f | sort | sed -n '2p')

make perf-cpu PERF_FRAME="$frame" PERF_REPS=20 PERF_THREADS=8 \
  PYTHON=.venv/bin/python
make checkpoint-oracle PERF_FRAME="$frame" PYTHON=.venv/bin/python
make tui-video TUI_DATA=/data/nuscenes/pointpillars_10sweep \
  PYTHON=.venv/bin/python
```

Keep fixture hashes and output boundaries with every number. Comparing this
real ten-sweep frame to the synthetic M2 regression or the WSL GPU fixture as
if they were the same workload would erase the very evidence the reports are
designed to preserve.
