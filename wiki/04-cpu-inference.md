# CPU Inference

The CPU backend is the complete reference backend. It owns the full graph and
does not depend on a third-party tensor runtime. Apple Accelerate/BNNS is an
optional per-operator adapter; unsupported shapes stay on the C path.

## Memory contract

The 23.19 MiB model mapping is read-only. A call uses bounded activation
storage:

| Buffer | Capacity |
|---|---:|
| PFN pillar output | up to 7.32 MiB |
| two stage buffers | 32.00 MiB |
| deblock/concat arena | 24.00 MiB |
| shared and middle scratch | 8.00 MiB |

Deblock outputs are written directly into consecutive slices of the shared
input. This avoids a separate concatenate allocation and copy. Stage tensors
ping-pong because each convolution kills its predecessor. Repeated CPU modes
own one grow-only, 64-byte-aligned workspace; single-process warm inference
therefore avoids allocator work without hiding retained memory or global state.

The first stride-two BEV convolution reads live pillar features through the
voxelizer's direct grid map. It never materializes or clears the 64 MiB dense
scatter tensor, and absent cells contribute implicit zeros. `PP_CPU_DENSE_FIRST=1`
restores scatter plus the generic dense convolution for differential checks.

## Kernels

The generic NCHW convolution is the portable and differential reference.
AVX2/FMA kernels widen over adjacent X positions and reuse each input vector
across four or eight output channels. Stride-two kernels share one gather
across the same output group. Plain head kernels accumulate a complete output
in registers before writing once.

Environment switches keep narrower/reference routes available:

| Switch | Route |
|---|---|
| `PP_CPU_DENSE_FIRST=1` | materialized scatter and dense first convolution |
| `PP_CPU_OC4=1` | four-output stride-one kernel |
| `PP_CPU_S2OC4=1` | four-output stride-two kernel |
| `PP_CPU_PLAIN_ACCUM=1` | memory-accumulating plain convolution |
| `PP_CPU_PLAIN_OC{1,2,4}=1` | cap direct plain output grouping |
| `PP_APPLE_DISABLE=1` | portable CPU path on macOS |

`tests/test_cpu_conv.c` compares padding edges, odd shapes, both strides,
ReLU, and plain output against a deterministic scalar oracle. `make
portable-test` rebuilds with `OMP=0` to keep the non-OpenMP route healthy.

Performance claims must come from `tools/perf.py` on a named real frame.
Thread count is part of the result because kernel topology and OpenMP overhead
change with both shape and host.
