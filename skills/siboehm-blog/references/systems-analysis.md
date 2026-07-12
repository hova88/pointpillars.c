# Systems Analysis Checklist

## Contents

1. Contract and stage inventory
2. CPU analysis
3. CUDA analysis
4. I/O and pipeline analysis
5. Evidence ledger

## 1. Contract and stage inventory

For every stage, record:

| Field | Questions |
|---|---|
| Input/output | Shape, dtype, layout, dynamic bounds? |
| Ownership | Who allocates, mutates, reuses, and frees it? |
| Residency | Host DRAM, page cache, CPU cache, GPU DRAM, shared memory, registers? |
| Work | FLOPs, comparisons, transcendentals, indexing, synchronization? |
| Traffic | Compulsory bytes and avoidable bytes? |
| Parallelism | Unit of independent work and scheduling granularity? |
| Correctness | Oracle, tolerance, malformed input, capacity behavior? |

Calculate peak live memory from overlapping lifetimes, not the sum of source-level arrays.

## 2. CPU analysis

Inspect each hot loop for:

- Address sequence: contiguous, fixed stride, gather, scatter, or pointer chasing.
- Cache-line utilization: useful bytes per 64-byte line and write-allocate effects.
- Reuse distance: registers, L1, private L2, shared LLC, or DRAM.
- Loop order: which dimension is contiguous and which tensor is reused.
- SIMD: alignment, trip count, reductions, masked edges, gathers, and compiler evidence.
- Threading: work ownership, false sharing, static/dynamic schedule, nested regions, and launch overhead.
- NUMA and page behavior when allocations become large.
- Branches and special functions in high-cardinality loops.

Relate optimizations to a mechanism. Examples: output-channel grouping reuses one input vector; sparse clearing scales writes with live pillars; logit thresholding removes unnecessary `expf` calls.

## 3. CUDA analysis

Inspect:

- Warp-to-output mapping and whether adjacent lanes touch adjacent addresses.
- Global transaction coalescing and alignment.
- L2/L1 reuse versus explicit shared-memory tiling.
- Shared-memory footprint, bank conflicts, barriers, and block residency.
- Register accumulation, occupancy limits, and instruction-level parallelism.
- Tensor-core fragment layout, padding, and shape quantization.
- Materialized transforms such as im2col: bytes written, bytes reread, and reuse count.
- Kernel launch count, stream ordering, events, and device-wide synchronization.
- Host/device boundary: pageable versus pinned behavior, compact transfers, and overlap.

Use a roofline-style argument only after calculating arithmetic intensity for the actual stage. Treat profiler access restrictions as missing evidence; use switchable A/B paths and CUDA events as alternatives.

## 4. I/O and pipeline analysis

- Distinguish cold file latency, page-cache-warm latency, preprocessing, inference, decode, rendering, and serialization.
- Draw a time sequence showing what overlaps and what blocks.
- Bound producer/consumer queues and include their memory cost.
- Preserve ordering and propagate worker errors.
- Measure throughput and single-item latency separately.

## 5. Evidence ledger

Label claims in working notes:

- **Measured:** command, fixture, repetitions, statistic, hardware.
- **Calculated:** formula and source dimensions.
- **Observed in source:** clickable file and symbol.
- **Inference:** plausible interpretation that still needs falsification.
- **External:** direct primary-source link.

Before publication, remove unsupported numerical claims or label them explicitly as estimates.
