# Publishing the technical article

The site is dependency-free and GitHub Pages serves the `docs/` artifact built
by `.github/workflows/pages.yml`. The checked JSON and Markdown are generated;
the HTML never parses a checkpoint in the browser.

## Preview

From the repository root:

```sh
make site-check
python3 -m http.server 8000 --directory docs
```

Open <http://localhost:8000>. Do not open `index.html` directly: browsers block
the local `fetch("model-data.json")` request under the `file:` scheme.

## Refresh evidence

Export the model and produce all five reports on one named fixture:

```sh
make model PYTHON=.venv/bin/python
make perf-cpu PERF_FRAME="$frame" PERF_REPS=20
make perf-cuda PERF_FRAME="$frame" PERF_REPS=20
make perf-cuda-compact PERF_FRAME="$frame" PERF_REPS=20
make perf-cudnn PERF_FRAME="$frame" PERF_REPS=20
make perf-cudnn-compact PERF_FRAME="$frame" PERF_REPS=20
make site-data
make site-check
```

`tools/build_site_data.py` validates the PPW header, tensor-table CRC, every
tensor CRC, the frozen config markers, and a shared checkpoint/point-cloud hash
across reports. It then writes:

- `docs/model-data.json`, the interactive site's machine-readable source;
- `docs/model-summary.md`, the same model inventory for JavaScript-free readers.

The source configuration is pinned to OpenPCDet commit
`233f849829b6ac19afb8af8837a0246890908755`. Keep raw performance reports in
`build/perf/`; they intentionally remain build artifacts. The compact published
bundle includes the hashes and protocol needed to identify the measurement.

## Evidence rules

- Keep cold latency separate from the warm median and p95.
- Compare routes only when model and fixture hashes match.
- Label custom FP16/WMMA as approximate; use `PP_CUDA_PRECISE=1` for the strict
  CUDA checkpoint oracle.
- Treat cuDNN FP32/FMA as strict only after its checkpoint oracle passes.
- Never turn one-frame latency or graph equivalence into a nuScenes mAP/NDS
  claim. Dataset accuracy requires the official evaluation loop.

Pushes to `main` that change the site automatically deploy to GitHub Pages. The
workflow runs the static-site checks before uploading the artifact.
