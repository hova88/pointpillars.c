# Validation

Validation is deliberately split into different claims:

1. Container and operator fixtures check bounds, shapes, and kernel indexing.
2. The checkpoint oracle compares end-to-end raw outputs for identical points
   and weights.
3. Decode fixtures check canonical boxes and rotated NMS.
4. Backend comparison checks CPU and any explicitly requested accelerator
   independently.

These are numerical and functional checks. This repository does not carry an
official dataset-metrics pipeline or claim that graph equivalence is a dataset
score.

## Local gates

```sh
make test
make portable-test
make checkpoint-oracle PERF_FRAME=/path/to/frame.bin
```

`make test` covers model mapping, voxelization, decode, CPU convolution, and
the TUI. The oracle uses the original checkpoint and deterministic
preprocessing. A mismatched checkpoint, configuration, point stride, sensor
frame, or sweep preparation is an invalid comparison and should fail loudly.

For CUDA work, build and test the requested backend separately. Approximate
WMMA and strict FP32 paths must not borrow one another's equivalence claims.
