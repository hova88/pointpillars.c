# The Frozen Contract: Specialization Before Optimization

> **Outcome.** PointPillars.c does not parse a training graph at runtime. It implements one exported model contract directly: ten-sweep nuScenes input, 11 pillar features, a fixed 512×512 BEV grid, three backbone scales, six multi-head groups, and ten classes. This rigidity is what makes every later memory calculation possible.

## Input geometry

The authoritative configuration is [`cfgs/pointpillars.yaml`](../cfgs/pointpillars.yaml). Each prepared point is five little-endian `float32` values:

```text
[x, y, z, intensity / 255, sweep_time_lag]
```

The accepted range is `[-51.2, -51.2, -5.0]` to `[51.2, 51.2, 3.0)`. A voxel is `0.2 × 0.2 × 8.0 m`, hence the complete horizontal grid is exactly `512 × 512 × 1`. The runtime keeps at most 30,000 non-empty pillars and 20 points per pillar.

The fifth value in a raw nuScenes lidar record is a ring index, not time. [`tools/prepare_nuscenes.py`](../tools/prepare_nuscenes.py) constructs the real contract: it transforms up to ten sweeps into the keyframe lidar coordinate system, filters near-ego points, normalizes intensity, computes time lag, and deterministically shuffles the result.

> **Sidenote — a five-float file can still be the wrong file.** Shape validation cannot distinguish ring index from time lag. Dataset/model pairing is semantic validation, not merely a byte-count check.

## Pillar feature contract

[`src/voxel.c`](../src/voxel.c) expands each accepted point to 11 values:

| Channels | Meaning |
|---|---|
| 0–4 | absolute `x, y, z, intensity, time_lag` |
| 5–7 | offset from the mean point in this pillar |
| 8–10 | offset from the geometric pillar center |

The maximum feature arena is `30,000 × 20 × 11 × 4 B = 25.18 MiB`. That is capacity, not per-frame useful data. The reference frame uses 7,868 pillars, so only about 6.60 MiB of pillar feature rows are live.

## Network shape map

The PFN maps each point's 11 values to 64 channels and max-reduces the 20 point
slots. The CPU backend feeds those `64 × P` live values directly into the first
convolution through the voxel grid map; CUDA materializes the equivalent
`64 × 512 × 512` NCHW canvas.

| Stage | Layers | Output | Notes |
|---|---:|---|---|
| PFN | linear + folded BN + ReLU + max | `P × 64` | `P ≤ 30,000` |
| Sparse BEV input | direct grid lookup / scatter | `64 × P` CPU, `64 × 512 × 512` CUDA | absent cells are zero |
| Backbone 0 | stride 2 + 3 stride-1 convs | `64 × 256 × 256` | 4 convolutions |
| Backbone 1 | stride 2 + 5 stride-1 convs | `128 × 128 × 128` | 6 convolutions |
| Backbone 2 | stride 2 + 5 stride-1 convs | `256 × 64 × 64` | 6 convolutions |
| Deblocks | 2× downsample, 1× identity, 2× upsample | three `128 × 128 × 128` | concatenated to 384 channels |
| Shared | 3×3 conv | `64 × 128 × 128` | common head input |
| Heads | 6 groups × 6 branches | 236 raw channels | each branch has mid + out conv |

Counting one multiply-accumulate as two floating-point operations gives:

| Group | GMAC | GFLOP |
|---|---:|---:|
| Backbone | 36.239 | 72.478 |
| Deblocks | 2.953 | 5.906 |
| Shared convolution | 3.624 | 7.248 |
| Multi-head branches | 23.970 | 47.941 |
| **Total convolution** | **66.786** | **133.572** |

These are derived from the live layer shapes in [`src/infer_cpu.c`](../src/infer_cpu.c), not from a hardware profiler. PFN, activations, indexing, and NMS add work outside this count.

## Why 236 output channels?

The six groups contain class counts `[1, 2, 2, 1, 2, 2]`. Each group owns six branches: classification, planar regression, height, size, sine/cosine angle, and velocity. [`pp_branch_channels`](../src/infer_cpu.c) computes their exact sizes. Summed over every head and branch, the raw output is:

```text
236 × 128 × 128 = 3,866,624 floats = 14.75 MiB
```

The class order and anchor sizes are frozen in [`src/decode.c`](../src/decode.c) and match the YAML. Moving one class between heads would change channel interpretation even if every tensor retained a valid shape.

## Export artifacts versus runtime semantics

[`tools/export_checkpoint.py`](../tools/export_checkpoint.py) does the framework-dependent work offline:

- folds BatchNorm into convolution weights and biases;
- uses epsilon `1e-3` for PFN/backbone and `1e-5` for SingleHead layers;
- renames 190 tensors into stable runtime names;
- emits contiguous little-endian fp32 arrays with 64-byte alignment and CRC32.

The runtime therefore executes convolution, activation, max reduction, sparse
BEV input or scatter, decode, and NMS. It does not need YAML parsing, PyTorch,
BatchNorm state, or plugin dispatch in the hot path.

## What to remember

- Fixed shapes are not a limitation accidentally tolerated by the engine; they are the optimization boundary.
- A model contract includes coordinate frames and feature meaning, not only tensor dimensions.
- Offline fusion reduces runtime operators and turns layout choices into a versioned file-format decision.

## What remains

Supporting another checkpoint requires a new verified contract. The safe sequence is exporter changes, tensor-shape fixtures, operator oracle comparison, raw graph comparison, then decoded and dataset-level validation.
