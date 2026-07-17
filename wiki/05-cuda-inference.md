# The CUDA Path: Auditable WMMA and Strict cuDNN

> **Outcome.** Two CUDA builds share one runtime contract. The dependency-free custom backend reaches 44.397 ms raw and 44.104 ms compact with shape-specific FP16 WMMA. The optional cuDNN backend keeps FP32/FMA graph equivalence and reaches 12.993 ms raw or 12.160 ms compact. Both retain same-binary fallbacks, bounded workspaces, device-resident activations, and an honest cold/warm profile.

![Hybrid implicit and explicit CUDA convolution](assets/cuda-hybrid.svg)

*The custom path uses implicit tiles for backbone and final outputs, but reuses one explicit transform across 36 middle heads. cuDNN consumes the same resident FP32 NCHW/OIHW buffers through cached shape plans.*

## One orchestration layer, two convolution providers

[`src/infer_cuda.cu`](../src/infer_cuda.cu) owns the fixed execution graph, persistent activation buffers, PFN/scatter, raw-output layout, compact candidate path, five timing events, and custom kernels. [`src/infer_cudnn.cu`](../src/infer_cudnn.cu) is a narrow optional provider for forward and transposed convolution.

```sh
make cuda     # no cuDNN dependency; custom WMMA default
make cudnn    # cuDNN FMA default; PP_CUDNN_DISABLE=1 falls back to custom
```

Both binaries upload the 23.19 MiB FP32 model once. Custom CUDA also creates an 11.59 MiB FP16 mirror. cuDNN consumes FP32 weights directly and therefore does not allocate that mirror or the custom head im2col arena.

The model pointer is locked to the persistent context. A second mapped model cannot silently reuse device addresses from the first.

## Residency before arithmetic

Three deblock destinations alias consecutive thirds of `cat`. Their final NCHW layout is already the `384×128×128` shared-convolution input, so no separate concat buffer or 24 MiB device copy exists.

Raw capacities on the perf fixture are:

| Backend | Capacity | Extra compact capacity |
|---|---:|---:|
| custom WMMA | 228.46 MiB | +32.50 MiB |
| cuDNN FMA | 206.97 MiB | +32.50 MiB |

The compact candidate arena is lazy and bounded at `10×65,536×52` bytes plus counters. It is allocated only by detect/batch/TUI paths.

## PFN and scatter

PFN assigns one thread to `(pillar, output_channel)`, loops over 20 points and 11 features, applies the folded affine transform and ReLU, and retains a local maximum. Scatter assigns threads across `64×P` and writes into a 64 MiB canvas cleared by `cudaMemsetAsync`.

Separate events now measure PFN and scatter instead of reporting a combined interval as PFN with a misleading zero scatter value. Warm medians are about `0.11 ms` PFN and `0.083 ms` scatter on the reference frame.

## Custom convolution: avoid global im2col where reuse is local

A 3×3 convolution is conceptually:

```text
W[co, ci×9] × Col[ci×9, h×w] → Y[co, h×w]
```

For the shared `384→64 @ 128²` layer, a global FP16 `Col` tensor would be 108 MiB. The custom implicit kernels instead assemble one 16×16 activation tile and padded weight tiles in shared memory, then execute WMMA with FP32 accumulators.

The original four-warp block covers 64 output channels. `wmma_conv_implicit8` uses eight warps so one activation tile feeds 128 output channels when `co≥128`. A same-binary 20-run A/B measured about 51.39 ms versus 53.16–54.09 ms for the four-warp fallback. `PP_CUDA_WARP4=1` restores it.

A 16-warp/256-output experiment did not improve end-to-end time and remains opt-in as `PP_CUDA_WARP16=1`.

## Middle heads: explicit transform reuse is cheaper

All 36 middle heads consume the same `64×128×128` shared feature. Their common transform is:

```text
576 × 16,384 fp16 = 18 MiB
```

The custom path materializes this matrix once and reuses it for all 36 middle convolutions. Rebuilding implicit tiles independently for every consumer lost the cross-branch reuse. This is the important distinction: global materialization is wasteful for one consumer, but can be useful for 36 consumers.

## Final heads: one warp and no repeated 18 MiB transform

Final branches produce only 2–12 output channels. The generic four-warp kernel prepared unused tiles for a 64-channel block. `wmma_conv_implicit_small` uses one warp and a single padded 16×16 output tile.

Final outputs also stopped materializing their own 18 MiB im2col. Middle heads retain the shared explicit transform; final outputs use the small implicit kernel. Same-binary A/B reduced total from about 51.2–51.5 ms to 48.49 ms, reduced head time from about 13.6 to 10.8 ms, and removed 18 MiB. `PP_CUDA_GENERIC_SMALL=1` and `PP_CUDA_EXPLICIT_OUTPUTS=1` restore the older routes.

## Custom precise mode and its accuracy boundary

`PP_CUDA_PRECISE=1` replaces WMMA convolution with FP32 warp reductions. Eight warps per block each own one output element and reduce input-channel products with shuffle operations. It is slow by design but passes the PyTorch checkpoint oracle at `6.523e-4` maximum error.

The default custom FP16-WMMA path is an approximate task-accuracy backend. Its optimized variants are byte-stable against their fallbacks, but the perf-frame comparison to PyTorch reports approximately `0.786` maximum and `0.006` mean raw error and changes one score-threshold-edge decoded box. It must not be described as graph-equivalent merely because the precise path passes.

## cuDNN forward plans

`make cudnn` compiles the optional provider against the installed cuDNN. On first use it creates descriptors and selects the fastest successful deterministic algorithm that fits `PP_CUDNN_WORKSPACE_MIB` (64 MiB by default). Plans are cached by shape, ReLU/identity mode, and math mode rather than tensor name, so repeated head shapes reuse descriptors and algorithms.

Input, output, FP32 weights, and bias point directly to persistent runtime buffers. ReLU layers use fused convolution+bias+activation. Plain final outputs use the same fused API with identity activation; `PP_CUDNN_SEPARATE_BIAS=1` restores separate convolution and bias calls for A/B.

The shared workspace grows only while cold plans are created and then remains fixed. The measured graph needed a 206.97 MiB total raw capacity, 21.49 MiB less than custom CUDA.

## cuDNN transposed convolution was the decisive second step

Replacing only forward convolutions reduced total latency from custom CUDA to about 31.6 ms, but the two custom FP32 deconvolutions then dominated. Their stride equals kernel size, so transposed convolution maps naturally to cuDNN backward-data convolution using the exported `[input_channel, output_channel, k, k]` weights.

`cudnnConvolutionBackwardData` plus bias and ReLU reduced the warm backbone/shared interval to about 6.28 ms. The final whole raw path reaches 12.993 ms, and compact detect reaches 12.160 ms. [`tests/test_cudnn.cu`](../tests/test_cudnn.cu) checks this mapping against a scalar oracle and observes zero error on its deterministic fixture.

NVIDIA's [cuDNN documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/index.html) describes tuned forward and backward convolution primitives. This runtime uses the installed 8.9 legacy C API because it preserves the repository's narrow C/CUDA ABI; the base `make cuda` target remains library-independent.

## FMA is the default; TF32 is an explicit approximation

The cuDNN path defaults to `CUDNN_FMA_MATH` with FP32 tensors. It passes the direct PyTorch oracle at `4.997e-4` maximum and `5.36e-6` mean absolute error. Two independent processes produced byte-identical raw output.

`PP_CUDNN_TF32=1` enables tensor-op conversion. On this graph it produced roughly `0.092` maximum and `8.74e-4` mean raw error, failed the declared allclose gate, and used about 22.6 MiB more workspace without a decisive latency advantage. It remains available for explicit task-accuracy experiments, not as the default.

## CUDA Graphs were measured, not assumed

Nsight Systems observed roughly 130 custom kernel launches per frame and a median `cudaLaunchKernel` API duration around `4.624 µs`. That justified a CUDA Graph experiment.

The fixed dense graph can be captured after PFN/scatter. Custom graph A/B was about 45.13 versus 45.09 ms; cuDNN graph A/B was about 13.33 versus 13.37 ms. Neither beat noise convincingly. They remain opt-in as `PP_CUDA_GRAPH=1` and `PP_CUDNN_GRAPH=1`.

Graph mode reports one combined dense interval in `backbone_ms` and zero `heads_ms`, because an event embedded within the captured graph is not an honest independent host-stage boundary. Default non-graph mode retains separate backbone/shared and head events.

## Grouping all six branch-middle convolutions also lost end to end

Concatenating six middle-head weight sets into one `64→384` convolution reduced measured head device time by about 0.3 ms. It also required packed weights and a `384×128×128` middle buffer, adding about 25 MiB, while total mean remained flat. `PP_CUDNN_GROUP_HEADS=1` preserves the experiment without making it the default.

## Stream ordering and five events

H2D copies, canvas clear, kernels, and events use one ordered stream. Default execution uses the legacy stream; graph experiments use the per-thread stream required for capture. Five events bracket:

1. PFN start;
2. PFN end;
3. scatter end;
4. backbone/shared end;
5. heads end.

No stage-level device synchronization is needed. `PP_CUDA_SYNC_STAGES=1` restores barriers for diagnosis. The final raw copy or compact candidate copy synchronizes the frame; `total_ms` is host-observed and includes the declared output boundary.

## Measured stage split

Twenty-run warm medians:

| Path | PFN | Scatter | Backbone/shared | Heads | Total |
|---|---:|---:|---:|---:|---:|
| custom raw | 0.110 | 0.083 | 32.200 | 10.217 | 44.397 ms |
| custom compact | 0.110 | 0.082 | 32.200 | 10.431 | 44.104 ms |
| cuDNN FMA raw | 0.113 | 0.083 | 6.284 | 4.058 | 12.993 ms |
| cuDNN FMA compact | 0.113 | 0.084 | 6.283 | 4.053 | 12.160 ms |

The raw cuDNN comparison is `-70.73%` total, `-80.48%` backbone/shared, `-60.28%` heads, and `-21.49 MiB` capacity. Compact is `-72.43%` total. PFN/scatter are intentionally unchanged.

Cold cuDNN is more expensive: the final measured first raw call was 301.0 ms versus 154.8 ms custom, because cuDNN creates plans and loads kernels. Interactive startup and sustained throughput therefore remain separate claims.

## Optimization and fallback map

| Mechanism | Default | Fallback / experiment |
|---|---|---|
| custom implicit backbone | yes | `PP_CUDA_EXPLICIT=1` |
| eight-warp wide blocks | yes | `PP_CUDA_WARP4=1` |
| 16-warp block | no | `PP_CUDA_WARP16=1` |
| one-warp final heads | yes | `PP_CUDA_GENERIC_SMALL=1` |
| implicit final outputs | yes | `PP_CUDA_EXPLICIT_OUTPUTS=1` |
| custom FP32 reference | no | `PP_CUDA_PRECISE=1` |
| custom dense Graph | no | `PP_CUDA_GRAPH=1` |
| cuDNN provider | `make cudnn` | `PP_CUDNN_DISABLE=1` |
| cuDNN strict FMA | yes | `PP_CUDNN_TF32=1` |
| deterministic algorithms | yes | `PP_CUDNN_NONDETERMINISTIC=1` |
| 64 MiB plan cap | yes | `PP_CUDNN_WORKSPACE_MIB=N` |
| cuDNN dense Graph | no | `PP_CUDNN_GRAPH=1` |
| grouped head mids | no | `PP_CUDNN_GROUP_HEADS=1` |

## What to remember

- Residency and layout aliasing matter before kernel selection.
- Explicit versus implicit transforms are chosen by consumer reuse, not a universal rule.
- A tuned library can expose the next custom bottleneck; forward cuDNN made deconvolution optimization necessary.
- Faster math modes require their own oracle and task-accuracy claims.
- Launch-count arguments still need end-to-end Graph measurements.

## What remains

Nsight Compute counters were unavailable under WSL (`ERR_NVGPUCTRPERM`), so occupancy, cache-hit, and shared-bank explanations remain hypotheses. cuDNN frontend graph fusion, multi-frame execution, and architecture-specific custom kernels are valid future experiments only if they beat the strict 12.160 ms compact baseline under the same protocol.
