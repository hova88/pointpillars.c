# Article evidence ledger

This file records what the article numbers mean. It is not a cross-platform
performance promise.

## Frozen artifacts

- Model: `nuscenes_multihead.ppw`
- PPW SHA-256: `ffbd7307bc98310a96859713570c448dcc375b4c5dbc6be45eabad9079eeaad4`
- Point fixture SHA-256: `0596eb2c7bb6ccf1af3817972416979567a187719b99276bc9cf9912ae63fc2d`
- Fixture: 26,414 input points, 23,741 accepted points, 7,854 live pillars,
  2,424 points clipped by the per-pillar cap, zero pillars dropped
- Model contract: 190 tensors, 6,073,319 fp32 values, 97 grouped entries,
  93 learned operators, 66.786 dense-convolution GMAC
- Architecture source: [OpenPCDet `cbgs_pp_multihead.yaml`](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml)

## Fresh latency reports

Recorded 21 July 2026 under WSL2 on an Intel i5-14600KF (16 logical CPUs) and
an NVIDIA RTX 4060 Ti 16 GB. Each row is one cold inference followed by 19 warm
inferences on the same frame.

| Route | Cold | Warm median | Warm p95 | Workspace | D2H |
|---|---:|---:|---:|---:|---:|
| CPU OpenMP/AVX2 raw | 547.285 ms | 504.051 ms | 585.131 ms | 65.92 MiB | 0 |
| custom CUDA raw | 171.920 ms | 47.591 ms | 51.099 ms | 228.46 MiB | 14.75 MiB |
| custom CUDA compact | 161.097 ms | 46.405 ms | 47.038 ms | 260.96 MiB | 19.84 KiB |
| cuDNN FP32/FMA raw | 634.380 ms | 14.161 ms | 15.469 ms | 206.97 MiB | 14.75 MiB |
| cuDNN FP32/FMA compact | 273.781 ms | **12.313 ms** | 14.490 ms | 239.47 MiB | 19.89 KiB |

The `stages_ms` values in `model-data.json` are warm medians of the runtime's
instrumented PFN, scatter, backbone, and head intervals. End-to-end latency also
contains voxelization, transfer, decode, NMS, and synchronization; stage values
therefore should not be summed and presented as the total.

## Correctness oracles

Against the original PyTorch/OpenPCDet checkpoint on the same real frame:

| Runtime path | Maximum absolute error | Mean absolute error | Result |
|---|---:|---:|---|
| CPU strict | 0.0010023117 | 1.06061e-5 | allclose |
| custom CUDA precise | 0.0006523132 | 5.77661e-6 | allclose |
| cuDNN FP32/FMA | 0.0004997253 | 5.36374e-6 | allclose |

The default custom CUDA FP16/WMMA route is deliberately an approximate speed
path (about 0.786 maximum raw error in the audited run). Its latency is useful;
it is not evidence of frozen-graph equivalence. The reference decoder produced
109 boxes at the documented score and rotated-NMS settings.

## Reproduction boundary

Use `make checkpoint-oracle`, `make checkpoint-oracle-cuda`, and
`make checkpoint-oracle-cudnn` for graph comparisons. Use OpenPCDet's official
nuScenes evaluation flow for mAP/NDS. The repository does not infer dataset
accuracy from tensor allclose or a single visually plausible frame.
