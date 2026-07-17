# Performance Workflow: Measure, Change, Prove

> **Outcome.** This is the step-by-step optimization record and the reusable performance protocol. Every promoted change has an identical model/frame contract, a visible cold run, a warm latency distribution, capacity and transfer accounting, a fallback switch, and a correctness gate at the appropriate level.

## Reference environment and final result

The controlled fixture is the lexicographically first prepared nuScenes frame: 26,414 points, 23,741 accepted points, and 7,854 live pillars. The measured host is an Intel i5-14600KF exposed as 16 logical CPUs by WSL; the GPU is an RTX 4060 Ti. Reports use 20 repetitions and discard only run zero from warm statistics.

| Backend and output contract | Threads | Warm median | p95 | Capacity | D2H |
|---|---:|---:|---:|---:|---:|
| native CPU raw | 16 | 468.816 ms | 487.872 ms | 129.92 MiB | 0 |
| GGML hybrid CPU raw | 16 | 458.973 ms | 474.963 ms | 147.92 MiB | 0 |
| native CPU raw, host-tuned | 32 | 410.406 ms | 419.656 ms | 129.92 MiB | 0 |
| custom WMMA CUDA raw | — | 44.397 ms | 46.814 ms | 228.46 MiB | 15,466,496 B |
| custom WMMA CUDA compact detect | — | 44.104 ms | 46.296 ms | 260.96 MiB | 20,320 B |
| cuDNN FMA raw | — | 12.993 ms | 13.536 ms | 206.97 MiB | 15,466,496 B |
| cuDNN FMA compact detect | — | 12.160 ms | 12.738 ms | 239.47 MiB | 20,372 B |

The 32-thread CPU result is a measured oversubscription result on this WSL topology, not a portable default. The compact CUDA result includes candidate compaction, compact D2H, canonical CPU decode, and rotated NMS. Raw mode instead transfers every output plane and exists primarily for graph-equivalence testing.

### Apple Silicon reference

A separate Apple M2 experiment uses a deterministic 24,000-point,
7,881-pillar fixture and the same raw-output boundary. Ten strict-hybrid runs
discard three warmups; the portable baseline uses three runs and discards run
zero.

| Apple M2 path | Warm median | p95 | Oracle max abs | Decision |
|---|---:|---:|---:|---|
| portable C | 7017.255 ms | 7029.027 ms | `6.75e-5` | strict baseline |
| BNNS convolution only, C deconvolution | 596.381 ms | 596.779 ms | — | diagnostic |
| strict BNNS/C hybrid | 249.490 ms | 253.699 ms | `6.51e-5` | promoted |
| all BNNS including 2×2/s2 | 181.041 ms | 186.170 ms | `2.12` | rejected |

The promoted route is `28.1×` faster than the portable baseline. It caches
3×3 and transposed-convolution filters, rotates the `[Cin,Cout,K,K]` deblock
weights once for BNNS' `IOHrWr` view, and leaves the numerically unsafe 2×2/s2
operator on canonical C. The diagnostic rows cannot be compared with the WSL
fixture or cited as task-level accuracy.

## Step 1: freeze the execution contract

Do not optimize an ambiguous graph. Pin the checkpoint, YAML, exporter, tensor order, voxel rules, score threshold, NMS threshold, and output boundary first.

```sh
make model
frame=$(find /data/nuscenes/pointpillars_10sweep \
  -name '*.bin' -type f | sort | head -1)
```

The perf report hashes the model, point file, and executable. It also records point/pillar statistics. A comparator refuses a changed model, fixture, machine, environment, repetition protocol, thread count, or output mode.

One frame is a micro-regression fixture. It is not a dataset throughput or task-accuracy claim; final accuracy changes still require the named nuScenes evaluation split.

## Step 2: build identified backend variants

```sh
make                     # dependency-free CPU
make ggml                # CPU plus pinned ggml v0.16.0
make cuda                # custom CUDA/WMMA
make cudnn               # FP32 cuDNN backend plus custom fallback
```

The Makefile hashes effective CPU/CUDA build configuration into the real binary and CUDA object paths, then refreshes stable `build/pointpillars*` names. Tests rebuild with the current flags every time. This prevents an `OMP=0` artifact from masquerading as an OpenMP performance build after `make portable-test`.

GGML is pinned to tag `v0.16.0` and commit `524f974bb21a1013408f76d71c15732482c0c3fe`. Its CMake directory is keyed by the absolute source path so a cache created from a different clone cannot be reused accidentally.

## Step 3: tune the host topology before tuning kernels

Thread count is part of the algorithm. Sweep it rather than inheriting a shell default:

```sh
for threads in 12 16 20 24 28 32 36; do
  OMP_NUM_THREADS=$threads \
    ./build/pointpillars bench nuscenes_multihead.ppw "$frame" 8
done
```

On the reference host, the native path moved from 468.816 ms at 16 workers to 410.406 ms at 32, then regressed at 36 in the sweep. The GGML path had a similar optimum, but the backend ordering changed: GGML won by 2.10% at 16 workers, while native C won by 1.20% at 32. This is why the runtime does not hard-code a machine-specific “GGML below N threads” heuristic.

`tools/perf.py` records `OMP_NUM_THREADS`, binding/placement variables when present, CPU identity, logical CPU count, governor, GPU identity, driver, and linked accelerator libraries. A comparison fails if relevant runtime environment differs.

## Step 4: declare the output boundary

CUDA has two meaningful protocols:

```sh
# Full 236-plane raw tensor: numerical oracle boundary.
./build/pointpillars_cuda bench-cuda nuscenes_multihead.ppw "$frame" 20

# Score-qualified compact candidates plus canonical decode/NMS.
./build/pointpillars_cuda bench-detect-cuda \
  nuscenes_multihead.ppw "$frame" 20
```

Raw mode transfers 15,466,496 bytes. Compact mode transferred about 20 KiB on this fixture. Comparing a raw baseline to a compact candidate would attribute an output-contract change to a kernel, so the JSON protocol stores `output_mode` and rejects that comparison.

## Step 5: capture cold and warm JSON reports

```sh
make perf-cpu          PERF_FRAME="$frame" PERF_REPS=20 PERF_THREADS=16
make perf-ggml         PERF_FRAME="$frame" PERF_REPS=20 PERF_THREADS=16
make perf-cuda         PERF_FRAME="$frame" PERF_REPS=20
make perf-cuda-compact PERF_FRAME="$frame" PERF_REPS=20
make perf-cudnn        PERF_FRAME="$frame" PERF_REPS=20
make perf-cudnn-compact PERF_FRAME="$frame" PERF_REPS=20
```

Each report contains every run and warm min/mean/median/p95/max/stdev for total, PFN, scatter, backbone/shared, and heads. It also records capacity bytes, D2H bytes, hashes, Git state, machine identity, linked libraries, and every CPU/CUDA/cuDNN A/B switch.

Run zero stays visible. Custom CUDA cold time includes context creation, device allocation, weight upload, and FP16 conversion. cuDNN additionally creates descriptors, selects algorithms, grows its bounded workspace, and loads library kernels. GGML creates fixed-shape graphs and work plans. Warm statistics exclude the configured warmup count; cold and warm are never averaged together.

## Step 6: form a falsifiable hypothesis

Before editing, write down:

1. the dominant measured stage;
2. the bytes, instructions, transforms, launches, transfers, or barriers to remove;
3. the exact shapes expected to win and those likely to regress;
4. the numerical tolerance and task-level gate;
5. the fallback switch and portability cost;
6. expected RAM/VRAM and D2H change.

This turns “try a faster library” into a test. GGUF is a mmap-friendly model container, not a kernel. GGML is a tensor runtime whose kernels can win or lose by shape. cuDNN is a tuned primitive library, but its algorithm, math mode, workspace, cold plan cost, and output numerics still belong in the experiment.

## Step 7: establish operator oracles before widening kernels

```sh
make test
make cudnn-test
```

`test_cpu_conv` compares stride-1, stride-2, padding-edge, ReLU, and plain-output paths against deterministic scalar C. Representative 64-channel fixtures stay below `5.96e-7` maximum error. `test_cudnn` checks fused ReLU and identity/bias convolution plus backward-data transposed convolution; measured maximum errors are `8.94e-8`, `8.94e-8`, and `0`.

These tests localize an indexing error before a 3.8-million-float graph comparison has to explain it. The CUDA small-head WMMA kernel did catch a real K-index bug this way: the full oracle initially reported a catastrophic error instead of allowing a fast but wrong kernel to survive.

## Step 8: optimize CPU reuse from the inside out

The CPU sequence was:

1. alias three deblock destinations directly into the final `384×128×128` shared-input arena; this removed a 24 MiB copy and allocation;
2. widen stride-1 ReLU convolution from four to eight output channels per loaded AVX2 input vector;
3. widen stride-2 gather convolution to eight output channels while collapsing `(channel_block,row)` for enough OpenMP tasks;
4. replace tiny final-head accumulation passes with one-, two-, four-, and eight-output direct register reductions, writing each result once;
5. retain scalar/generic and narrower same-binary fallbacks for every specialization.

Useful switches are `PP_CPU_OC4`, `PP_CPU_S2OC4`, `PP_CPU_PLAIN_ACCUM`, and `PP_CPU_PLAIN_OC{1,2,4}`. The final 16-thread native warm median is 468.816 ms with a 129.92 MiB workspace.

### GGML/GGUF decision

The fixed `.ppw` file already has mmap ownership, aligned FP32 tensors, offline BatchNorm folding, and no tokenizer or dynamic metadata. Re-encoding it as GGUF would change packaging, not convolution cost. The official [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) likewise describes a model storage format designed for loading, extensibility, and mmap compatibility.

GGML v0.16.0 F32 convolution was benchmarked per representative shape. It lost on the wide first stride-2 layer and stride-1 layers, but won on `64→128 @ 256²/s2` and `128→256 @ 128²/s2`. `src/infer_ggml.c` therefore owns only those two shapes, points tensors directly at existing NCHW/OIHW memory, caches two fixed graphs, and uses one bounded work buffer. `PP_GGML_DISABLE=1` restores native C.

At 16 workers this hybrid improves total median from 468.816 to 458.973 ms and adds 18 MiB. At the host-tuned 32 workers it regresses from 410.406 to 415.312 ms, so the globally fastest measured CPU configuration remains native C with 32 workers.

Fake post-training quantization was rejected. Weight-only INT8 and activation-plus-weight INT8 screens produced mean raw errors around `0.31–0.43` and maxima around `78–80`; neither passed the graph oracle. No low-bit path was promoted without calibration and task-accuracy evidence.

## Step 9: optimize the dependency-free CUDA path

The custom CUDA sequence was:

1. remove the 24 MiB deblock concat copy;
2. build implicit activation tiles on chip for backbone/shared convolution;
3. reuse one 18 MiB explicit im2col across all 36 middle heads;
4. share one activation tile across eight warps/128 output channels for `co≥128` (`PP_CUDA_WARP4=1` restores four warps);
5. use one-warp implicit WMMA for final heads with at most 12 output channels (`PP_CUDA_GENERIC_SMALL=1` restores the generic block);
6. stop materializing an 18 MiB im2col for every final output convolution (`PP_CUDA_EXPLICIT_OUTPUTS=1` restores it);
7. split five CUDA events across PFN, scatter, backbone/shared, and heads without adding stage barriers.

The final raw workspace is 228.46 MiB and warm median is 44.397 ms. The precise `PP_CUDA_PRECISE=1` FP32 reduction path remains the custom graph-equivalence reference; the default FP16-WMMA path is deliberately approximate.

## Step 10: integrate cuDNN as an optional strict backend

A representative-shape probe showed FP32 cuDNN forward convolution around 0.1–1.0 ms per layer, so the experiment justified integration rather than speculation. `src/infer_cudnn.cu` then added:

- descriptors and algorithms cached by shape, not tensor name;
- direct use of the existing FP32 device model and NCHW/OIHW activations—no FP16 mirror or repack per frame;
- deterministic algorithms within a default 64 MiB workspace cap;
- fused convolution+bias+ReLU or identity for forward layers;
- `cudnnConvolutionBackwardData` for the two non-overlapping transposed convolutions;
- one shared workspace that grows only during cold plan creation.

NVIDIA describes cuDNN as a tuned library for convolution and other DNN primitives, and documents forward and backward-data convolution in its official [cuDNN documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/index.html). The repository intentionally uses the installed 8.9 legacy C API because it matches the narrow C/CUDA runtime boundary; `make cuda` remains independent of cuDNN.

Default cuDNN math is strict FP32 `CUDNN_FMA_MATH`. It reaches 12.993 ms raw and 12.160 ms compact, passes the checkpoint oracle at `4.997e-4` maximum error, and uses 21.49 MiB less capacity than custom CUDA. `PP_CUDNN_DISABLE=1` falls back to the custom path in the same binary, and that fallback is byte-identical to `build/pointpillars_cuda`.

`PP_CUDNN_TF32=1` is opt-in. On the fixture it used about 22.6 MiB more workspace and failed the raw oracle (`max_abs≈0.092`, `allclose=False`) without a decisive total-latency advantage, so it is not the default.

## Step 11: compare only compatible reports

```sh
python3 tools/perf.py compare \
  build/perf/cuda.json build/perf/cudnn.json \
  --stat median --max-regression 0 \
  --max-workspace-growth-mib 0

python3 tools/perf.py compare \
  build/perf/cuda-compact.json build/perf/cudnn-compact.json \
  --stat median --max-regression 0 \
  --max-workspace-growth-mib 0 --max-d2h-growth-mib 0.1
```

The final raw comparison is `-70.73%` total, `-80.48%` backbone, `-60.28%` heads, and `-21.49 MiB` capacity. Compact is `-72.43%` total, `-80.49%` backbone, and `-61.14%` heads with the same capacity reduction. The comparator exits nonzero on a latency, workspace, or transfer gate and refuses mismatched identities unless exploratory `--allow-mismatch` is explicit.

## Step 12: run the correctness funnel

```sh
make test
make portable-test
make cudnn-test
make checkpoint-oracle
make checkpoint-oracle-ggml
make checkpoint-oracle-cuda
make checkpoint-oracle-cudnn
```

The final graph-oracle maxima are:

| Path | Maximum absolute error | Result |
|---|---:|---|
| native CPU | `9.890e-4` | allclose |
| GGML hybrid | `9.632e-4` | allclose |
| custom precise CUDA | `6.523e-4` | allclose |
| cuDNN FMA | `4.997e-4` | allclose |

The custom FP16-WMMA path is not allowed to borrow this claim: on the perf frame it had about `0.786` maximum and `0.006` mean raw error and changed one decoded threshold-edge box. It is an evaluated approximate/task-accuracy path, while CPU, precise CUDA, and cuDNN FMA are graph-equivalence paths.

Compact cuDNN and full-raw cuDNN produced byte-identical JSON on the controlled frame. Existing custom compact/full testing covers 81 mini frames. Accuracy-sensitive changes still run `make evaluate` on the documented model/split pairing.

The final cuDNN FMA batch processed all 404 prepared frames and scored the official 81-frame `mini_val`: mAP `0.205512`, NDS `0.327992`. The checked-in custom FP16-WMMA run is mAP `0.20557`, NDS `0.32804`; the strict backend therefore preserves the task-level result within the observed evaluator precision.

ASan/UBSan builds cover model, voxel, decode, CPU convolution, and TUI fixtures. The final clean build also passes warning-clean `make test`, `make portable-test`, custom CUDA, cuDNN, cuDNN fixtures, and pinned GGML rebuilds.

## Step 13: preserve negative results

| Experiment | Evidence | Decision / switch |
|---|---|---|
| GGUF-only conversion | `.ppw` already mmap/aligned; no kernel change | rejected as a performance claim |
| INT8 fake PTQ | large raw error, no oracle pass | rejected |
| GGML for every conv | loses on several shapes and at 32 workers | two-shape hybrid only |
| custom 16-warp WMMA | no win over eight warps | opt-in `PP_CUDA_WARP16=1` |
| custom dense CUDA Graph | about 45.13 vs 45.09 ms | opt-in `PP_CUDA_GRAPH=1` |
| cuDNN dense CUDA Graph | about 13.33 vs 13.37 ms | opt-in `PP_CUDNN_GRAPH=1` |
| grouped six-branch cuDNN mids | head -0.3 ms, total flat, +25 MiB | opt-in `PP_CUDNN_GROUP_HEADS=1` |
| cuDNN TF32 | oracle failure, +22.6 MiB | opt-in `PP_CUDNN_TF32=1` |
| pinned D2H staging | extra host memcpy regressed total | removed |

Nsight Systems observed roughly 130 kernel launches per custom frame and a median `cudaLaunchKernel` API duration around `4.624 µs`. Nsight Compute hardware counters were unavailable under WSL (`ERR_NVGPUCTRPERM`), so no occupancy, cache-hit, or bank-conflict number is claimed without evidence.

## Step 14: promotion checklist

Before changing a default:

1. save the before JSON outside `build/`;
2. change one mechanism and keep a same-binary fallback;
3. run the smallest scalar/operator fixture;
4. run the full raw oracle at the declared tolerance;
5. compare identical warm distributions, capacity, and transfer bytes;
6. validate compact/full decoded identity when the boundary changes;
7. run portable and sanitizer gates;
8. run task evaluation for any approximate numerical path;
9. document cold cost and rejected alternatives;
10. promote only the configuration that wins end to end.
