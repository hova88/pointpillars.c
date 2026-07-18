# Local nuScenes Mini Workflow

The runtime expects prepared little-endian `[N,5]` float32 files. After
extracting nuScenes mini under `/data/nuscenes`, create those files with:

```sh
make prepare-data PYTHON=.venv/bin/python
```

`tools/prepare_nuscenes.py` transforms up to ten lidar sweeps into the
keyframe lidar coordinate system, filters near-ego points, normalizes
intensity, computes time lag, and writes deterministic frame files plus a
manifest.

Run one frame or the viewer:

```sh
frame=$(find /data/nuscenes/pointpillars_10sweep \
  -name '*.bin' -type f | sort | head -1)

./build/pointpillars infer nuscenes_multihead.ppw "$frame" result.ppout 5 boxes.json
./build/pointpillars tui nuscenes_multihead.ppw \
  /data/nuscenes/pointpillars_10sweep
```

Use the same frame for a performance report and checkpoint-oracle comparison:

```sh
make perf-cpu PERF_FRAME="$frame" PERF_REPS=20 PERF_THREADS=8
make checkpoint-oracle PERF_FRAME="$frame" PYTHON=.venv/bin/python
```

This workflow validates preparation, inference, decode, and visualization. It
does not install a dataset devkit or convert detections into an official
submission schema.
