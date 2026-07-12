# End-to-End Pipeline: Latency, Throughput, and Peak Memory

> **Outcome.** Network latency is only one lane of the system. Batch mode overlaps next-frame loading and voxelization with current-frame CUDA work through a two-slot producer. CUDA keeps dense workspaces resident, uses events rather than stage barriers, and transfers compact candidates. The result is bounded memory, ordered output, and explicit switches for every speculative optimization.

![Overlapped CPU and GPU execution timeline](assets/execution-timeline.svg)

*The producer advances one frame ahead while a single CUDA stream stays ordered by dependencies and measured by events.*

## CLI as an execution router

[`src/main.c`](../src/main.c) provides inspect, infer, benchmark, batch, and TUI modes with CPU/CUDA variants. The CLI is intentionally thin around the same runtime primitives:

```text
load → voxelize → infer → decode → serialize/render
```

Single-frame `infer-cuda` preserves full 14.75 MiB raw output because its `.ppout` file is an oracle artifact. Batch and TUI default to `pp_infer_cuda_detect`, where compact candidates replace the full D2H copy. One API should not silently weaken another API's output contract in the name of speed.

## The bounded producer

Batch mode creates `prep_pipe` with one or two `prep_slot` values. A slot owns a persistent `pp_pillars`, stats, frame index, preparation time, state, and error text. The worker:

1. waits until its modulo-selected slot is empty;
2. loads one file and voxelizes into that slot;
3. publishes state `ready` under a mutex and condition variable;
4. stops on the first error.

The consumer waits for frames in sorted filename order, runs inference and output, then marks the slot empty. Two slots bound lookahead to one frame and preserve deterministic ordering. `PP_NO_PREFETCH=1` reduces depth to one for memory-constrained systems or A/B.

The additional host cost is roughly one 27 MiB pillar workspace. In a same-binary 20-frame A/B, two-slot prefetch improved median wall time by about 2.6%; the exact gain varies with GPU clock and filesystem cache state.

## Host residency

Each active pillar slot owns:

- maximum feature arena: 25.18 MiB;
- coordinates: 0.46 MiB;
- point counts: 0.03 MiB;
- persistent grid: 1 MiB.

Point-file storage is exact-size and freed after voxelization. The decoded output allocation holds at most 1,000 `pp_box` entries in the CLI. Compact CUDA mode does not allocate the 14.75 MiB host raw-output arena for batch/TUI.

## Device residency

![Persistent host and device memory](assets/memory-residency.svg)

*Dense CNN weights and activation arenas stay resident; only sparse frame inputs and compact detections cross the boundary.*

The important live device allocations are:

| Allocation | MiB |
|---|---:|
| fp32 model | 23.19 |
| fp16 model mirror | 11.59 |
| max input features | 25.18 |
| pillar output | 7.32 |
| scatter | 64.00 |
| stage ping-pong | 32.00 |
| three upsample outputs | 24.00 |
| concatenated feature | 24.00 |
| shared + middle | 8.00 |
| raw heads | 14.75 |
| head im2col workspaces and smaller buffers | remainder |
| **hybrid inference total** | **about 270.5** |
| optional compact capacity | **+32.5** |

This accounting is capacity-based and excludes CUDA runtime/allocator reserve. It is still more honest than quoting model size alone: activation and transform workspaces dominate the 23 MiB checkpoint.

## Synchronization boundaries

All kernels use the default CUDA stream, so data dependencies already impose order. Four events record PFN start/end, backbone end, and heads end. Stage times come from event deltas after the final transfer synchronizes the frame.

Earlier code used `cudaDeviceSynchronize` after every stage. Removing those host-visible barriers had a small and noisy throughput benefit, but it also makes the execution model cleaner: stage measurement no longer changes stage scheduling. `PP_CUDA_SYNC_STAGES=1` restores the old behavior.

## Cold versus warm

Cold CUDA includes device allocation, page commitment, fp32 upload, fp16 conversion, and lazy compact allocation. A representative first call can be hundreds of milliseconds while warm calls are around 55–58 ms. Both matter:

- interactive startup cares about cold latency;
- sustained batch/TUI throughput cares about warm latency;
- benchmark reports should never average them without saying so.

## Transfer boundary

Full raw output always transfers 15,104 KiB. Compact detection averaged roughly 10.5 KiB on 81 mini frames. The reduction is over three orders of magnitude, yet end-to-end speedup is modest because convolution dominates and pageable D2H was only a few milliseconds.

A persistent pinned staging experiment regressed slightly because it added an explicit host memcpy; it was removed. This negative result matters: a mechanism that improves bus bandwidth can still lose after counting the extra copy.

## Error propagation

- Allocation and thread initialization failures unwind owned slots.
- Worker load/voxel errors become slot state `error` and stop production.
- Malformed `.bin` size, empty directories, and non-`.bin` suffixes are rejected.
- Explicit CUDA requests fail when CUDA is unavailable.
- Batch exits nonzero on preparation, inference, decode, or serialization failure.

## What to remember

- Optimize throughput with a timeline, not a sum of isolated stage times.
- Queue depth is also a memory budget; two slots are enough to overlap one producer with one consumer.
- Persistent activation memory, not model size, sets the accelerator footprint.

## What remains

Multiple CUDA streams are unlikely to overlap dependent network stages, but H2D copy of a prepared next frame could overlap late current-frame work with pinned input buffers. That requires measuring the extra host-copy and registration costs that already defeated pinned output staging.
