# The CPU Path: Cache Lines Before Clever Intrinsics

> **Outcome.** The complete C backend evaluates the same 133.57-GFLOP graph in 591.99 ms on 16 threads of an i5-14600KF. The important choices are NCHW ownership by output channel, eight-row L1 working sets, four-channel AVX2 reuse, static OpenMP scheduling, and a scalar boundary path that keeps the vector interior simple.

![AVX2 input reuse and output ownership](assets/cpu-locality.svg)

*Eight adjacent x positions are reused across four output-channel accumulators; generic row tiles keep partial outputs close to L1.*

## First account for memory

The CPU workspace in [`pp_infer_cpu`](../src/infer_cpu.c) contains:

| Buffer | Size on the 7,868-pillar fixture |
|---|---:|
| PFN pillar output `64 × P` | 1.92 MiB |
| Scatter `64 × 512 × 512` | 64 MiB |
| Two stage buffers | 32 MiB |
| Three upsample buffers | 24 MiB |
| Concatenated feature | 24 MiB |
| Shared + middle scratch | 8 MiB |
| **Total** | **about 153.9 MiB** |

The 23.19 MiB model mapping and 14.75 MiB raw output live outside that reported workspace. The stage buffers ping-pong because consecutive convolutions do not need older activations. This is lifetime planning in plain C: two buffers replace sixteen layer-specific allocations.

## PFN: small reduction, awkward layout

OpenMP assigns independent `(pillar, output_channel)` pairs. For each pair, the loop evaluates 20 points × 11 features and retains the maximum after ReLU. Input points for one pillar are contiguous, but output is stored channel-major as `pillar[oc * P + pi]` because scatter and later NCHW operations consume channel planes.

The feature weights for one output channel are only 44 bytes and remain hot. The point records are 44 bytes each, sequentially visited. PFN is not the dominant stage: after warmup it is around 2 ms in the measured CPU run.

## Scatter: sparse writes into a 64 MiB canvas

Each OpenMP iteration owns one output channel, then walks every pillar. Writes within a channel plane jump to `(y, x)` locations determined by the shuffled points. There is no false sharing between output-channel owners, but spatial writes have poor locality. The preceding `calloc` guarantees zero for empty cells and pays the cost of touching a dense 64 MiB result because the following convolution is dense anyway.

## Generic convolution: keep partial outputs in L1

The fallback `conv3_relu` parallelizes over output channels. Within one output channel it processes eight output rows at a time:

```text
8 rows × 256 columns × 4 B = 8 KiB
```

At the widest post-stride layer, that partial output tile fits in L1 while the code loops over all input channels and nine kernel taps. The innermost `ox` traversal is contiguous for stride one and marked `omp simd`. Weights for one `(oc, ic)` pair are nine floats; input rows stream through cache lines.

This loop order makes a deliberate trade: it reloads input data for different output channels, but avoids writing and rereading a full output plane for every input channel. Each OpenMP worker owns its output plane, so there is no reduction or lock.

## Specialized stride-one kernel: four channels per input vector

When AVX2 is available and output channels are divisible by four, `conv3s1_relu4` changes the reuse unit. One unaligned `__m256` loads eight adjacent input pixels. Four broadcast weights update four independent vector accumulators using FMA.

For the same input vector, the scalar conceptual path would load it once per output channel. Grouping four channels amortizes that load and keeps 32 output results in registers (`4 × 8`). ReLU becomes four vector max operations at the end of the reduction.

Interior x positions use this vector path. Boundary pixels use a scalar branch, which removes padding checks from the hot vector loop. The boundary cost is proportional to the perimeter, not the image area.

## Stride two: gather is the honest cost

Adjacent outputs consume input x positions separated by two. `conv3s2_relu4` therefore uses `_mm256_i32gather_ps` with indices `[0,2,4,…,14]`. Gather is less friendly than a contiguous load: it asks hardware to collect values from multiple positions and may consume more load-port resources. It is still useful because four output channels share the gathered vector.

This is a good example of a shape-specific limit. Stride-one convolution can use full cache-line locality; stride-two convolution cannot pretend its logical neighbors are physically contiguous.

## Deblocks and heads

The stage-0 deblock is a non-overlapping 2×2 stride-two convolution. The other deblocks are transposed convolutions whose stride equals kernel size, so output tiles do not overlap. That property removes atomic accumulation and lets each output channel own a plane.

The shared feature is `64 × 128 × 128`. Every one of 36 branch-middle convolutions reads it, then an output convolution produces a small branch-specific channel count. This repeated 3×3 work explains why heads account for roughly 188–192 ms of the CPU run despite their compact final output.

## Measured stage split

Reference fixture, 16 threads, four warm runs:

| Stage | Typical warm latency |
|---|---:|
| PFN | 2.1–2.5 ms |
| Backbone + deblocks | 392–404 ms |
| Shared convolution + heads | 187–192 ms |
| Total warm mean | 591.99 ms |

At 133.57 GFLOP, the effective graph rate is about 226 GFLOP/s. The number is useful for comparing this exact graph, but it does not imply every CPU instruction is an FMA; gathers, address arithmetic, ReLU, allocation, and cache misses are included in latency.

The CPU profile boundary closes `backbone_ms` before concat/shared; the CUDA event boundary closes it after shared. Compare total latency directly, but account for this boundary difference before comparing backend stage labels.

## Portability

`make portable-test` rebuilds with `OMP=0`. Unknown pragmas are suppressed, the scalar/vector code remains valid, and CUDA stays optional. `-march=native` enables AVX2/FMA on the measured machine; a distributable binary should choose a lower baseline or add runtime dispatch.

## What to remember

- SIMD comes after loop order: first choose a working set and ownership model that caches can support.
- Grouping output channels is a data-reuse optimization, not merely "using AVX2."
- Static output-channel ownership avoids synchronization and false sharing, but load balance depends on channel counts and uniform spatial work.

## What remains

The CPU workspace is allocated per inference call. Persistent arenas could reduce allocation/page costs in batch use. More aggressive channel blocking or packed weights might improve L2 reuse, but any change should be measured separately for stride-one, stride-two, deblock, shared, and tiny output-head shapes.
