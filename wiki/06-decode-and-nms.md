# Decode, Compact Transfer, and Rotated NMS

> **Outcome.** Postprocessing converts 236 raw feature planes into at most 830 canonical boxes. CPU decode fell from roughly 2.6 ms to about 0.36 ms by comparing logits before sigmoid and prebinding branch planes. CUDA batch/TUI avoid transferring the 14.75 MiB raw output: they return about 10.5 KiB of original candidate codes and run the same C geometry and NMS.

## Score filtering before transcendental work

The raw classifier produces 36 score planes across the six heads. That is:

```text
36 × 128 × 128 = 589,824 logits
```

The original conceptual loop could evaluate sigmoid for every logit, then compare against score threshold `s`. Sigmoid is monotonic, so [`src/decode.c`](../src/decode.c) instead calculates the threshold once:

```text
logit_threshold = log(s / (1 - s))
```

Only logits above that value pay for `expf`. On normal frames, almost every location is rejected. This changes no accepted score and removes hundreds of thousands of transcendental calls.

Branch addresses are also resolved once per `(head, anchor, code)` rather than by scanning branch boundaries inside every accepted candidate.

## Residual box decode

Each candidate carries ten codes:

| Codes | Meaning |
|---|---|
| 0–1 | x/y offsets scaled by anchor diagonal |
| 2 | height offset scaled by anchor height |
| 3–5 | log dimensions, exponentiated and multiplied by anchor size |
| 6–7 | cosine/sine angle residual |
| 8–9 | x/y velocity |

Anchor centers span the 128×128 output grid over the `[-51.2, 51.2]` range. Each class has one configured size, bottom height, and two rotations. Angle reconstruction uses `atan2` on the predicted sine/cosine pair plus the anchor rotation.

The result is `pp_box`: position, dimensions, yaw, velocity, score, and class ID. JSON output, official evaluation, and TUI all consume this exact structure.

## Rotated IoU without an external geometry library

`iou` creates four corners for each rectangle, then clips one polygon against the four edges of the other using cross products and segment intersections. The clipped polygon has at most 16 points in the fixed stack arrays. Shoelace area gives intersection area; union is the two rectangle areas minus intersection.

Per class, candidates are sorted by score, truncated to `PRE_MAX=1000`, and greedily compared with accepted boxes until `POST_PER_CLASS=83`. The worst configured output is therefore `10 × 83 = 830` boxes.

This is quadratic in the retained candidate count, but aggressive score filtering and pre-max truncation bound it. A spatial index would add complexity for a stage currently around a fraction of a millisecond.

## Compact GPU candidates

![Host and device memory residency](assets/memory-residency.svg)

*The compact boundary retains original logits and regression codes, then returns to the canonical CPU decoder.*

For batch and TUI, [`compact_candidates`](../src/infer_cuda.cu) runs one class-specific kernel per class. Threads inspect logits in device-resident raw output. Accepted positions atomically reserve a slot and copy:

```text
10 regression floats + 1 logit + position + head/anchor/class metadata
= 52 bytes per pp_compact_candidate
```

The GPU does **not** decode box geometry. The CPU receives original fp32 values and calls `pp_decode_compact`, which performs the same arithmetic and `append_nms` as full-raw decode. That boundary was chosen for validation, not merely convenience.

Worst-case capacity is exact: a class can have at most four anchors × 128×128 positions = 65,536 candidates. Ten fixed segments therefore reserve 32.5 MiB on device, allocated lazily. Actual D2H on 81 checked mini frames averaged 10.54 KiB and peaked at 19.1 KiB, versus 15,104 KiB for every full raw frame.

`PP_CUDA_RAW_DECODE=1` disables compaction. All JSON files from compact and full-raw paths were byte-identical across those 81 frames.

> **Sidenote — atomic order.** Candidates are sorted by score before NMS, so reservation order does not define semantic output. Exact byte identity on the validation set is still checked rather than assumed.

## What to remember

- Move cheap monotonic comparisons before expensive special functions.
- Compact at the accelerator boundary, but retain original values when CPU arithmetic is the validated authority.
- A bounded quadratic algorithm can be the right endpoint when earlier filters make its real input small.

## What remains

A device-side top-k/NMS path could remove the remaining compact CPU work, but it would create a second geometry implementation and a more difficult equivalence surface. At current latency, that trade is not automatically favorable.
