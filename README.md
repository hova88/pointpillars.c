# pointpillars.c — nuScenes MultiHead

A specialized, dependency-free C runtime for the native OpenPCDet nuScenes PointPillars MultiHead checkpoint. It includes optimized AVX2/FMA CPU kernels, custom CUDA FP32 and Tensor Core paths, official nuScenes evaluation, and a live ANSI/Braille point-cloud TUI.

![cover](./docs/cover.png)

## Exact model contract

The authoritative inputs are:

-  [`ckpts/pp_multihead_nds5823_updated.pth`](https://drive.google.com/file/d/1p-501mTWsq0G9RzroTWSXreIMyTUUpBM/view?usp=sharing)
- `cfgs/pointpillars.yaml`
- `/data/nuscenes/v1.0-mini`

The runtime implements the checkpoint directly:

- 10-sweep input `[x, y, z, intensity/255, time_lag]`;
- range `[-51.2,-51.2,-5, 51.2,51.2,3]`, voxel `0.2×0.2×8`;
- at most 30,000 pillars and 20 points per pillar;
- 11-feature PFN, 512×512 scatter and three-stage BEV backbone;
- 384→64 shared convolution;
- six separate heads over ten nuScenes classes;
- separate `cls/reg/height/size/sincos-angle/velocity` branches;
- per-class rotated NMS with threshold 0.2.

## Build

Python and PyTorch are used only by the offline checkpoint converter. Inference uses C/POSIX APIs; OpenMP is optional. CUDA is an optional build target.

```sh
make model                 # native checkpoint -> 190-tensor, 23.19 MiB container
make                       # CPU runtime
make cuda                  # custom CUDA runtime
make test
make portable-test         # clean OMP=0 build and tests
```

Override `CHECKPOINT`, `CONFIG`, `NVCC`, or `CUDA_ARCH` as needed.

## Prepare nuScenes exactly

Raw nuScenes lidar files store ring index in the fifth float. The model instead requires sweep time. Build OpenPCDet-compatible ten-sweep frames once:

```sh
python3 tools/prepare_nuscenes.py \
  --root /data/nuscenes \
  --output /data/nuscenes/pointpillars_10sweep
```

The converter performs sweep-local ego filtering, sensor/ego/global transforms, intensity normalization, time-lag construction and deterministic test-time shuffling. The included mini split produces 404 frames and roughly 2 GB of point data.

## Inference and benchmarks

```sh
frame=$(find /data/nuscenes/pointpillars_10sweep -name '*.bin' | head -1)

OMP_NUM_THREADS=16 ./build/pointpillars infer \
  nuscenes_multihead.ppw "$frame" result.ppout 5 boxes.json

./build/pointpillars_cuda infer-cuda \
  nuscenes_multihead.ppw "$frame" result.ppout 5 boxes.json

# Strict FP32 CUDA graph-equivalence mode:
PP_CUDA_PRECISE=1 ./build/pointpillars_cuda infer-cuda \
  nuscenes_multihead.ppw "$frame" precise.ppout 5

./build/pointpillars_cuda bench-cuda nuscenes_multihead.ppw "$frame" 10
```

Directory batch inference overlaps loading and voxelization of the next frame
with the current inference using a bounded two-frame pipeline. Set
`PP_NO_PREFETCH=1` to disable overlap for memory-constrained systems or A/B
measurements; this saves one roughly 27 MiB host-side pillar workspace.

CUDA batch and TUI modes compact score-qualified candidates on the device and
run the canonical C box decode/NMS on their original logits and regression
codes. On the mini split this reduces device-to-host traffic from 15,104 KiB to
about 10.5 KiB per frame while remaining byte-identical to full-raw detection
output across 81 checked frames. `PP_CUDA_RAW_DECODE=1` restores full tensor
transfer for differential testing. The bounded compact buffer adds 32.5 MiB of
device capacity only when this path is used.

Measured on the current i5-14600KF / RTX 4060 Ti system with a 328k-point ten-sweep frame:

| backend | warm network latency |
|---|---:|
| CPU AVX2/FMA, 16 threads | ~603 ms |
| CUDA precise FP32 | ~604 ms |
| CUDA hybrid implicit/explicit WMMA fast path | ~55 ms |

The precise CUDA and CPU outputs match the direct PyTorch checkpoint oracle. The
fast path is explicitly approximate; the current reference-frame oracle reports
maximum absolute error `8.66e-4` and `allclose=True`.

The default CUDA path uses implicit WMMA for the backbone and a reused explicit
im2col for the multi-head branches. Set `PP_CUDA_EXPLICIT=1` for the older fully
explicit path. The hybrid path uses about 270.5 MiB of persistent device memory,
down from roughly 414.5 MiB for the explicit workspace.
Stage profiling uses CUDA events so PFN, backbone, and heads remain on one
continuous stream without host-side device barriers. Set
`PP_CUDA_SYNC_STAGES=1` to restore the older synchronization points for A/B or
driver diagnosis.

## Live terminal point cloud

```sh
./build/pointpillars_cuda tui-cuda \
  nuscenes_multihead.ppw /data/nuscenes/pointpillars_10sweep
```

Each terminal character carries a 2×4 Braille pixel block. Rotated boxes are overlaid in ANSI color. No curses or graphical window is required.
The viewer runs in an alternate screen and restores the terminal on exit. Press
`Space` to pause, arrows or `p`/`n` to step frames, `WASD` to pan, `z`/`e` to
rotate, `+`/`-` to zoom, `0`–`9` to toggle nuScenes classes, `,`/`.` to adjust
the score threshold, `[`/`]` to inspect detections, `b`/`v` to toggle box and
velocity layers, `r` to reset the view, and `q` to quit. Detection classes use
stable colors and the selected target shows pose, dimensions, yaw, velocity,
and confidence in a status panel. `g` toggles the 10 m metric grid and `t`
toggles bounded cross-frame trails. Trails use same-class nearest-neighbor
association and reset automatically after reverse navigation or a frame jump.
The selected-object panel reports its bounded local track ID and history age.
Paused views redraw on terminal resize; interrupt, quit, suspend, and hangup
signals all leave through the terminal-restoration path.

## Official nuScenes evaluation

```sh
rm -rf build/nuscenes-detections
./build/pointpillars_cuda batch-cuda nuscenes_multihead.ppw \
  /data/nuscenes/pointpillars_10sweep build/nuscenes-detections

python3 tools/make_submission.py \
  build/nuscenes-detections \
  /data/nuscenes/pointpillars_10sweep/manifest.json \
  evaluation/nuscenes_submission.json

python3 tools/evaluate_nuscenes.py \
  evaluation/nuscenes_submission.json \
  --output evaluation/nuscenes-mini
```

On the local 81-frame official `mini_val` split, the current fast path measures mAP `0.2056` and NDS `0.3280`. This must not be compared directly with the checkpoint filename's 58.23 NDS, which refers to a different/full evaluation split.

## Architecture and optimization

- Versioned, aligned, bounds-checked, per-tensor CRC32 model container.
- Offline BatchNorm folding with the correct `1e-3` backbone/PFN epsilon and `1e-5` SingleHead epsilon.
- AVX2/FMA four-output-channel CPU microkernels and stride-2 gathers.
- Persistent CUDA weights/workspaces and device-resident intermediates.
- FP16 implicit-im2col plus WMMA with FP32 accumulation; explicit precise fallback.
- MultiHead channel and anchor ordering reproduced from upstream OpenPCDet.
- Sine/cosine ResidualCoder, velocity decode and per-class rotated NMS.
- One canonical detection representation shared by JSON, official evaluation and TUI.

The Colibri-derived reasoning about residency, offline specialization, validation gates and measured optimization remains installed separately as the `colibri-code-mind` skill.

## Systems performance wiki

The illustrated [PointPillars.c performance wiki](wiki/Home.md) walks the entire
runtime from its frozen model contract through voxel locality, CPU caches,
CUDA tensor-core tiling, bounded pipeline residency, compact decode, validation,
and the interactive terminal renderer. The repository also ships the reusable
[`siboehm-blog`](skills/siboehm-blog/SKILL.md) skill used to produce and extend
that publication-ready documentation.
