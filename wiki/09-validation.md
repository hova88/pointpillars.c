# Validation: From Corrupt Bytes to nuScenes NDS

> **Outcome.** The repository keeps three claims separate: a kernel can match a fixture, a graph can match the checkpoint, and decoded boxes can score on a named dataset split. The validation funnel prevents a fast approximation from borrowing credibility from the wrong test.

![Validation funnel](assets/validation-funnel.svg)

*Each lower-level claim depends on the safety and equivalence evidence above it.*

## Focused C fixtures

`make test` builds small executables rather than a general test framework:

| Test | Contract |
|---|---|
| [`test_model.c`](../tests/test_model.c) | 190 records, representative tensor shapes, mapped container |
| [`test_voxel.c`](../tests/test_voxel.c) | clipping, features, coordinates, zero padding across reuse |
| [`test_decode.c`](../tests/test_decode.c) | empty output, one candidate, compact/full byte equality |
| [`test_tui.c`](../tests/test_tui.c) | track growth, jump reset, missed-frame expiry |

Exit codes identify failed assertions with almost no runtime dependency. `make portable-test` cleans and repeats the suite with OpenMP disabled.

Sanitizer builds cover model, voxel, and decode paths. Malformed tests include truncated model headers, invalid point-file byte counts, and empty input directories.

## Framework oracle

[`tools/oracle_checkpoint.py`](../tools/oracle_checkpoint.py) is a direct PyTorch execution of the exported checkpoint contract. It reimplements the exact voxel features, PFN, scatter, three backbone stages, deblocks, shared convolution, and all 36 head branches with TF32 disabled.

It reads `.ppout`, concatenates raw PyTorch outputs in runtime branch order, and reports max/mean absolute difference plus `np.allclose`. On the reference fixture, the current CUDA path reports:

```text
max_abs  = 8.659e-4
mean_abs = 1.441e-5
allclose = True   (rtol=2e-4, atol=2e-3)
```

Precise CUDA and CPU are also compared directly. The FP16-WMMA path is approximate; passing the declared tolerance does not imply bit identity with FP32.

## Decode equivalence

Compact CUDA transfer does not return raw planes, so its gate is decoded identity. Full-raw and compact modes were run on 81 mini frames and every generated JSON file compared byte-for-byte. This verifies candidate thresholding, code packing, CPU reconstruction, ordering, and NMS together on real data.

The switches make differential tests reproducible:

```sh
PP_CUDA_RAW_DECODE=1  # full 15,104 KiB raw D2H
PP_CUDA_EXPLICIT=1    # fully explicit im2col CUDA graph
PP_CUDA_PRECISE=1     # direct FP32 CUDA convolution
PP_CUDA_SYNC_STAGES=1 # device barriers at profile boundaries
```

## Official task accuracy

[`tools/make_submission.py`](../tools/make_submission.py) transforms lidar-frame centers, rotations, and velocities through calibrated sensor and ego poses into nuScenes global coordinates. It emits the official detection schema and motion attributes.

[`tools/evaluate_nuscenes.py`](../tools/evaluate_nuscenes.py) invokes the official nuScenes detection evaluator. The checked-in [`metrics_summary.json`](../evaluation/nuscenes-mini/metrics_summary.json) records on the local 81-frame `mini_val` split:

| Metric | Value |
|---|---:|
| mAP | 0.20557 |
| NDS | 0.32804 |
| translation error | 0.52881 |
| scale error | 0.53181 |
| orientation error | 0.61572 |
| velocity error | 0.56981 |

The checkpoint filename's `5823` does not describe this mini split. Comparing those values directly would mix datasets and evaluation contexts.

## Benchmark discipline

The CLI prints every run and excludes run zero from its warm mean. A publication claim should state:

- point count and live pillar count;
- CPU model, thread count, GPU model, CUDA architecture;
- precise or approximate backend and environment switches;
- cold call separately from warm statistic;
- raw or compact output boundary;
- correctness gate run on the same code.

Profiler access was unavailable for hardware counters in the WSL environment. The repository therefore uses CUDA events for stage timing and same-binary switches for causal A/B. It does not invent occupancy or cache-hit numbers.

## What to remember

- Numerical equivalence, decoded equivalence, and dataset accuracy answer different questions.
- A benchmark without its output boundary can hide 15 MiB of transfer or a CPU decode stage.
- A rejected optimization is still valuable when its switch and measurement falsify a plausible mechanism.

## What remains

The focused fixtures do not isolate every convolution shape. Saved per-operator oracle tensors would make future packed-weight or quantized-kernel work safer, especially for small output heads and edge padding.
