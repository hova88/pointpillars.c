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

Measured on the current i5-14600KF / RTX 4060 Ti system with a 328k-point ten-sweep frame:

| backend | warm network latency |
|---|---:|
| CPU AVX2/FMA, 16 threads | ~603 ms |
| CUDA precise FP32 | ~604 ms |
| CUDA FP16 WMMA fast path | ~76 ms |

The precise CUDA and CPU outputs match the direct PyTorch checkpoint oracle with maximum absolute error below `7.4e-4`. The fast path is explicitly approximate; on the reference frame it retains the same 66 decoded detections.

## Live terminal point cloud

```sh
./build/pointpillars_cuda tui-cuda \
  nuscenes_multihead.ppw /data/nuscenes/pointpillars_10sweep
```

Each terminal character carries a 2×4 Braille pixel block. Rotated boxes are overlaid in ANSI color. No curses or graphical window is required.

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
