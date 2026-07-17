#!/usr/bin/env python3
"""Reproducible PointPillars benchmark capture and regression comparison."""

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import statistics
import subprocess
import sys


RUN_RE = re.compile(
    r"run (?P<run>\d+) total (?P<total>[0-9.]+) ms \("
    r"pfn (?P<pfn>[0-9.]+), scatter (?P<scatter>[0-9.]+), "
    r"backbone (?P<backbone>[0-9.]+), heads (?P<heads>[0-9.]+), "
    r"workspace (?P<workspace>\d+), d2h (?P<d2h>\d+)\)"
)
POINT_RE = re.compile(
    r"points=(?P<points>\d+) accepted=(?P<accepted>\d+) "
    r"pillars=(?P<pillars>\d+) clipped=(?P<clipped>\d+) dropped=(?P<dropped>\d+)"
)
METRICS = ("total", "pfn", "scatter", "backbone", "heads")
IDENTITY_PATHS = ("model", "points")
CUDA_SWITCHES = (
    "PP_CUDA_EXPLICIT", "PP_CUDA_EXPLICIT_OUTPUTS", "PP_CUDA_GENERIC_SMALL", "PP_CUDA_LEGACY_IM2COL", "PP_CUDA_PRECISE",
    "PP_CUDA_GRAPH", "PP_CUDA_LEGACY_STREAM", "PP_CUDA_RAW_DECODE", "PP_CUDA_SYNC_STAGES", "PP_CUDA_WARP4", "PP_CUDA_WARP16",
    "PP_CUDNN_DISABLE", "PP_CUDNN_GRAPH", "PP_CUDNN_GROUP_HEADS", "PP_CUDNN_NONDETERMINISTIC", "PP_CUDNN_SEPARATE_BIAS",
    "PP_CUDNN_TF32", "PP_CUDNN_WORKSPACE_MIB",
)
CPU_SWITCHES = ("PP_APPLE_CONV2", "PP_APPLE_DISABLE", "PP_APPLE_DECONV_DISABLE", "PP_APPLE_MIN_SPATIAL", "PP_APPLE_PLAIN_DISABLE", "PP_CPU_OC4", "PP_CPU_PLAIN_ACCUM", "PP_CPU_PLAIN_OC1", "PP_CPU_PLAIN_OC2", "PP_CPU_PLAIN_OC4", "PP_CPU_S2OC4", "PP_GGML_DISABLE")
RUNTIME_ENV = (
    "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_DYNAMIC",
    "OMP_WAIT_POLICY", "CUDA_VISIBLE_DEVICES",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command):
    try:
        return subprocess.run(command, check=True, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def percentile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(values):
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def cpu_name():
    if pathlib.Path("/proc/cpuinfo").exists():
        for line in pathlib.Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        return command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or \
               command_output(["sysctl", "-n", "hw.model"]) or "Apple Silicon"
    return platform.processor() or "unknown"


def cpu_governor():
    path = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return path.read_text().strip()
    except OSError:
        return None


def linked_accelerators(binary):
    output = command_output(["otool", "-L", str(binary)]) if platform.system() == "Darwin" \
             else command_output(["ldd", str(binary)])
    if output is None:
        return []
    names = ("accelerate", "bnns", "cudnn", "cuda", "ggml", "gomp", "omp")
    return [line.strip() for line in output.splitlines()
            if any(name in line.lower() for name in names)]


def git_state():
    revision = command_output(["git", "rev-parse", "HEAD"])
    status = command_output(["git", "status", "--porcelain"])
    return {"revision": revision, "dirty": bool(status) if status is not None else None}


def run_benchmark(args):
    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).resolve()
    points = pathlib.Path(args.points).resolve()
    for path in (binary, model, points):
        if not path.is_file():
            raise SystemExit(f"not a file: {path}")
    if args.reps < 2:
        raise SystemExit("--reps must be at least 2 (one cold and one warm run)")
    if args.warmup < 1 or args.warmup >= args.reps:
        raise SystemExit("--warmup must be at least 1 and less than --reps")

    env = os.environ.copy()
    if args.backend == "cpu":
        env["OMP_NUM_THREADS"] = str(args.threads)
    if args.output_mode == "compact" and args.backend != "cuda":
        raise SystemExit("--output-mode compact requires --backend cuda")
    mode = ("bench-detect-cuda" if args.output_mode == "compact" else
            "bench-cuda" if args.backend == "cuda" else "bench")
    command = [str(binary), mode, str(model), str(points), str(args.reps)]
    completed = subprocess.run(command, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)

    runs = []
    fixture = None
    for line in completed.stderr.splitlines():
        match = RUN_RE.fullmatch(line)
        if match:
            item = {key: int(value) if key in ("run", "workspace", "d2h")
                    else float(value) for key, value in match.groupdict().items()}
            runs.append(item)
        point_match = POINT_RE.fullmatch(line)
        if point_match:
            fixture = {key: int(value) for key, value in point_match.groupdict().items()}
    if len(runs) != args.reps or fixture is None:
        sys.stderr.write(completed.stderr)
        raise SystemExit("benchmark output did not match the expected protocol")

    warm = runs[args.warmup:]
    if not warm:
        raise SystemExit("--warmup must leave at least one measured run")
    switch_names = CUDA_SWITCHES if args.backend == "cuda" else CPU_SWITCHES
    switches = {name: env.get(name) for name in switch_names if env.get(name) is not None}
    gpu = command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"])
    report = {
        "schema": "pointpillars-perf-v1",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "backend": args.backend,
        "command": command,
        "environment": {
            "omp_num_threads": args.threads if args.backend == "cpu" else None,
            "cpu_switches": switches if args.backend == "cpu" else {},
            "cuda_switches": switches if args.backend == "cuda" else {},
            "runtime": {name: env.get(name) for name in RUNTIME_ENV
                        if env.get(name) is not None},
        },
        "machine": {
            "platform": platform.platform(),
            "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "cpu_governor": cpu_governor(),
            "gpu": gpu,
        },
        "git": git_state(),
        "artifacts": {
            "binary": {"path": str(binary), "sha256": sha256(binary),
                       "accelerator_libraries": linked_accelerators(binary)},
            "model": {"path": str(model), "sha256": sha256(model)},
            "points": {"path": str(points), "sha256": sha256(points)},
        },
        "fixture": fixture,
        "protocol": {"repetitions": args.reps, "warmup_runs": args.warmup,
                     "output_mode": args.output_mode},
        "cold": runs[:args.warmup],
        "warm_runs": warm,
        "warm": {metric: summarize([item[metric] for item in warm]) for metric in METRICS},
        "workspace_bytes": max(item["workspace"] for item in runs),
        "device_to_host_bytes": max(item["d2h"] for item in runs),
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = pathlib.Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output)
    else:
        sys.stdout.write(output)
    summary = report["warm"]["total"]
    print(f"{args.backend}: cold {runs[0]['total']:.3f} ms, "
          f"warm median {summary['median']:.3f} ms, p95 {summary['p95']:.3f} ms",
          file=sys.stderr)


def compare_reports(args):
    baseline = json.loads(pathlib.Path(args.baseline).read_text())
    candidate = json.loads(pathlib.Path(args.candidate).read_text())
    if baseline.get("schema") != "pointpillars-perf-v1" or candidate.get("schema") != "pointpillars-perf-v1":
        raise SystemExit("both inputs must use schema pointpillars-perf-v1")
    mismatches = []
    if baseline["backend"] != candidate["backend"]:
        mismatches.append("backend")
    for name in IDENTITY_PATHS:
        if baseline["artifacts"][name]["sha256"] != candidate["artifacts"][name]["sha256"]:
            mismatches.append(name)
    baseline_environment = dict(baseline["environment"])
    candidate_environment = dict(candidate["environment"])
    for environment in (baseline_environment, candidate_environment):
        environment.setdefault("cpu_switches", {})
        environment.setdefault("cuda_switches", {})
        environment.setdefault("runtime", {})
    if baseline_environment != candidate_environment:
        mismatches.append("environment")
    if baseline["machine"] != candidate["machine"]:
        mismatches.append("machine")
    if baseline["fixture"] != candidate["fixture"]:
        mismatches.append("fixture")
    if baseline["protocol"] != candidate["protocol"]:
        mismatches.append("protocol")
    if mismatches and not args.allow_mismatch:
        raise SystemExit("incomparable reports (use --allow-mismatch to override): " + ", ".join(mismatches))

    failed = False
    print("metric       baseline   candidate      delta")
    for metric in METRICS:
        before = baseline["warm"][metric][args.stat]
        after = candidate["warm"][metric][args.stat]
        delta = (after / before - 1.0) * 100.0 if before else 0.0
        print(f"{metric:10s} {before:9.3f} {after:11.3f} {delta:+9.2f}%")
        if metric == "total" and delta > args.max_regression:
            failed = True
    print("capacity/transfer")
    for name in ("workspace_bytes", "device_to_host_bytes"):
        before = baseline[name]
        after = candidate[name]
        delta_mib = (after - before) / (1024 * 1024)
        print(f"{name:22s} {before:12d} {after:12d} {delta_mib:+9.2f} MiB")
        limit = (args.max_workspace_growth_mib if name == "workspace_bytes"
                 else args.max_d2h_growth_mib)
        if limit is not None and delta_mib > limit:
            failed = True
    if failed:
        print("FAIL: one or more regression gates failed", file=sys.stderr)
        return 1
    print(f"PASS: total {args.stat} regression is within {args.max_regression:.2f}%", file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="capture a benchmark report")
    run.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    run.add_argument("--binary", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--points", required=True)
    run.add_argument("--reps", type=int, default=10)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--threads", type=int, default=16)
    run.add_argument("--output-mode", choices=("raw", "compact"), default="raw")
    run.add_argument("--output")
    run.set_defaults(function=run_benchmark)
    compare = sub.add_parser("compare", help="compare compatible reports")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--stat", choices=("mean", "median", "p95", "min"), default="median")
    compare.add_argument("--max-regression", type=float, default=5.0)
    compare.add_argument("--max-workspace-growth-mib", type=float)
    compare.add_argument("--max-d2h-growth-mib", type=float)
    compare.add_argument("--allow-mismatch", action="store_true")
    compare.set_defaults(function=compare_reports)
    args = parser.parse_args()
    result = args.function(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
