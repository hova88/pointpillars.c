# PointPillars.c

PointPillars.c is a small C11 inference runtime for one frozen OpenPCDet
nuScenes PointPillars MultiHead checkpoint. Python is used only to export the
checkpoint, prepare ten-sweep input, run the checkpoint oracle, and summarize
performance. Runtime inference, decode, batch output, and the terminal viewer
do not require Python packages.

The runtime keeps one canonical contract:

```text
little-endian float32 points [N,5]
  -> bounded pillars [P,20,11]
  -> PFN and sparse-grid BEV input (CPU) / dense scatter (CUDA)
  -> NCHW backbone and six heads
  -> decoded pp_box values
```

The CPU backend is complete and portable. OpenMP/AVX2, Apple
Accelerate/BNNS, custom CUDA, and cuDNN are narrow acceleration paths with
explicit fallbacks or build gates. The model stays in a bounds-checked,
memory-mapped `.ppw` container.

## Repository map

| Area | Role |
|---|---|
| `src/model.c` | mapped model validation and tensor lookup |
| `src/voxel.c` | point loading and deterministic pillar creation |
| `src/infer_cpu.c` | full scalar/OpenMP/AVX2 CPU graph |
| `src/infer_apple.c` | optional Accelerate/BNNS adapter |
| `src/infer_cuda.cu` | optional custom CUDA backend |
| `src/infer_cudnn.cu` | optional strict-FP32 cuDNN backend |
| `src/decode.c` | canonical decode and rotated NMS |
| `src/tui.c` | event-driven ANSI/Braille viewer |
| `tools/` | export, input preparation, oracle, and performance reports |
| `tests/` | focused container, operator, decode, voxel, and TUI fixtures |

## Build and verify

```sh
make model
make
make test
make portable-test
```

CUDA and cuDNN remain opt-in through `make cuda` and `make cudnn`. An explicit
accelerator request fails if that backend is unavailable; it does not silently
pretend to be accelerated.

## Chapters

1. [Model contract](01-model-contract.md)
2. [Voxelization](02-voxelization.md)
3. [Model container](03-model-container.md)
4. [CPU inference](04-cpu-inference.md)
5. [CUDA inference](05-cuda-inference.md)
6. [Decode and NMS](06-decode-and-nms.md)
7. [Pipeline and memory](07-pipeline-and-memory.md)
8. [Terminal viewer](08-terminal-visualizer.md)
9. [Validation](09-validation.md)
10. [Extension guide](10-extension-guide.md)
11. [Performance workflow](11-performance-workflow.md)
12. [Apple Silicon](12-macos-apple-silicon.md)
13. [Local nuScenes mini workflow](13-local-macos-nuscenes-mini.md)
