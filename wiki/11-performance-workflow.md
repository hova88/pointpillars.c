# Performance Workflow

Optimize the measured end-to-end bottleneck on a named real frame. Keep run
zero as cold evidence and summarize warm runs separately.

```sh
frame=/data/nuscenes/pointpillars_10sweep/example.bin
make perf-cpu PERF_FRAME="$frame" PERF_REPS=20 PERF_THREADS=16
make perf-cuda PERF_FRAME="$frame" PERF_REPS=20
make perf-cuda-compact PERF_FRAME="$frame" PERF_REPS=20
make perf-cudnn PERF_FRAME="$frame" PERF_REPS=20
make perf-cudnn-compact PERF_FRAME="$frame" PERF_REPS=20
```

`tools/perf.py` records binary identity, environment switches, point/pillar
counts, stage timings, workspace, transfer bytes, and cold/warm statistics.
Its compare mode can reject a configured total-latency regression.

For every optimization record:

- the observed bottleneck and proposed mechanism;
- numerical tolerance and fallback route;
- peak workspace and host/device transfer;
- cold and warm latency at a fixed thread count;
- the exact model and input identity.

Operator microbenchmarks explain a result but do not replace full-pipeline
evidence. Quantization, isolated GPU offload, speculative prefetch, and new
libraries stay experiments until they improve the complete path without
weakening the declared correctness boundary.
