# Extension Guide: Change One Contract at a Time

> **Outcome.** The runtime is small because specialization replaces a general graph executor. Extensions should preserve that advantage: modify the offline contract, bind one narrow runtime path, add a differential gate, and measure the real shape rather than introducing abstraction before evidence.

## Add or change a model

1. Start in [`tools/export_checkpoint.py`](../tools/export_checkpoint.py).
2. Record the new preprocessing, classes, anchors, operators, tensor shapes, BN epsilons, and branch order.
3. Bump the container version if record or data semantics change.
4. Add representative shape assertions to [`tests/test_model.c`](../tests/test_model.c).
5. Update the PyTorch oracle before changing optimized kernels.
6. Make CPU raw output pass; then custom precise CUDA and cuDNN FMA; only then evaluate approximate CUDA math.
7. Revalidate decode and the correct dataset split.

Do not teach the runtime to parse an arbitrary training graph unless generality is an explicit product requirement. A second specialized forward function is often easier to audit than a half-built graph VM.

## Add a CPU kernel

Use the scalar/generic convolution as the semantic fallback. Dispatch only on properties that the implementation actually requires: stride, kernel, channel divisibility, alignment, and CPU feature.

For each candidate kernel, document:

| Question | Evidence |
|---|---|
| What bytes are reused? | cache-line and loop-order diagram |
| What stays in registers? | accumulator count and compiler output |
| What is the per-thread ownership? | no-overlap proof |
| What shapes win? | per-shape benchmark, not only full graph |
| What is the fallback? | portable build and scalar edge path |

Avoid global `-march=native` if distributing one binary across machines; add runtime CPUID dispatch or separate build artifacts.

## Add a CUDA kernel

Keep `pp_cuda.h` a narrow C ABI. Persistent state belongs to the CUDA context; CPU completeness must not depend on CUDA headers or runtime libraries.

Before coding, calculate:

- output tile and lane ownership;
- global transactions per tile;
- shared bytes and barriers per block;
- register accumulators per thread/warp;
- padding and inactive-lane cost for every target shape;
- whether a materialized transform has multiple consumers;
- additional persistent and peak VRAM.

Add an environment switch for A/B until the full graph proves a win. Preserve `PP_CUDA_PRECISE=1` as the graph-equivalence route. A fast kernel that only works on the largest backbone shape is not a replacement for the small heads.

For a cuDNN operator, cache descriptors and algorithms by the shape properties that determine execution, not by tensor name. Account for plan-creation cold time, selected workspace, determinism, math mode, and transposed-convolution layout. `PP_CUDNN_DISABLE=1` must keep the custom backend usable in the same binary. Reduced-precision modes require their own oracle and task evaluation.

## Change voxel features

Feature changes cross four boundaries:

1. [`tools/prepare_nuscenes.py`](../tools/prepare_nuscenes.py) point-file meaning;
2. [`src/voxel.c`](../src/voxel.c) feature construction and zero padding;
3. PFN weight shape in model export and both backends;
4. [`tools/oracle_checkpoint.py`](../tools/oracle_checkpoint.py) reference construction.

Update them together. A five-float input with the wrong fifth-channel meaning passes file-size validation and still produces invalid predictions.

## Add an output format

Consume `pp_detections`, not raw branch tensors. [`write_detections`](../src/main.c), submission conversion, and TUI demonstrate three consumers of the canonical representation. New consumers should not reimplement sigmoid, anchor decode, or NMS.

If raw output is required for an oracle or downstream research, use `pp_raw_output` and state the 236-plane order. Do not silently switch an API from full raw to compact candidates.

## Add a TUI layer

Add bounded state to `pp_tui_state`, update it once per decoded frame, and render from that state without rerunning inference on redraw. Assign a raster priority that composes predictably with grid, points, tracks, boxes, and selection.

New key handling must preserve:

- paused redraw;
- previous/next frame semantics;
- narrow-terminal layout;
- SIGWINCH behavior;
- termios restoration on normal and signal exits.

Add state-machine logic to [`tests/test_tui.c`](../tests/test_tui.c) and use a PTY integration test for terminal behavior.

## Add a pipeline stage

Draw its lifetime on the [execution timeline](07-pipeline-and-memory.md). Decide whether it belongs to producer, GPU stream, compact CPU decode, or consumer serialization. Bound any queue and include its buffers in peak memory.

Preserve frame order and propagate errors across threads. Throughput improvement does not excuse a hidden extra frame of latency or unbounded prefetch.

## Repository commands

```sh
make test             # normal CPU fixtures
make portable-test    # no OpenMP
make ggml             # pinned optional CPU hybrid
make cuda             # custom optional accelerator
make cudnn             # optional strict FP32/FMA accelerator
make cudnn-test        # scalar cuDNN operator fixtures
make checkpoint-oracle
make checkpoint-oracle-cudnn
make evaluate
```

Run the Wiki validator after documentation changes:

```sh
python3 skills/siboehm-blog/scripts/check_wiki.py wiki --repository-root .
```

## Primary external references

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA WMMA API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [NVIDIA cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/index.html)
- [GGML](https://github.com/ggml-org/ggml)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [CUTLASS efficient GEMM documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- [OpenMP specification](https://www.openmp.org/specifications/)
- [nuScenes detection task](https://www.nuscenes.org/object-detection/)
- [The optimization-worklog inspiration](https://siboehm.com/articles/22/CUDA-MMM)

## What to remember

- Extend contracts vertically—from exporter through oracle—not by patching one hot loop in isolation.
- Generality has a runtime and audit cost. Pay it only when the repository actually needs it.
- Every optimization needs a mechanism, an off switch during evaluation, a correctness gate, and an end-to-end measurement.

## What remains

The best next extension is the one supported by a real use case and a falsifiable bottleneck. The repository deliberately leaves hypotheses visible instead of promising that every available optimization should be implemented.
