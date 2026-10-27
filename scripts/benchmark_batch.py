"""Reproducible benchmark comparing sequential analyze() vs batched analyze_batch().

Usage:
    python scripts/benchmark_batch.py tests/sample_faces/happy_face.jpg
    python scripts/benchmark_batch.py path/to/img.jpg --batch-counts 5 10 20
    python scripts/benchmark_batch.py path/to/img.jpg --json results_batch.json

Measures end-to-end processing of image batches sequentially (looping analyze())
versus batched (analyze_batch()). Run on an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from erml import EmotionDetector


def percentile(values: list[float], percent: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percent))


def benchmark_run(
    detector: EmotionDetector,
    images: list[np.ndarray],
    warmup: int,
    iterations: int,
    mode: str,
    batch_size: int = 32,
) -> dict[str, object]:
    if mode == "sequential":

        def run_fn() -> None:
            for img in images:
                detector.analyze(img)

    elif mode == "batched":

        def run_fn() -> None:
            detector.analyze_batch(images, batch_size=batch_size)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    for _ in range(warmup):
        run_fn()

    durations_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        run_fn()
        durations_ms.append((time.perf_counter() - started) * 1000)

    return {
        "mode": mode,
        "image_count": len(images),
        "iterations": iterations,
        "warmup_iterations": warmup,
        "latency_ms": {
            "min": round(min(durations_ms), 3),
            "median": round(statistics.median(durations_ms), 3),
            "p95": round(percentile(durations_ms, 95), 3),
            "max": round(max(durations_ms), 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=Path("tests/sample_faces/happy_face.jpg"),
        help="path to a face image (default: tests/sample_faces/happy_face.jpg)",
    )
    parser.add_argument(
        "--batch-counts",
        nargs="+",
        type=int,
        default=[2, 5, 10, 20],
        metavar="COUNT",
        help="number of images per batch to evaluate (default: 2 5 10 20)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="internal ONNX inference chunk size for analyze_batch (default: 32)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="timed runs per test (default: 15)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="warm-up runs per test (default: 3)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="optional path to save complete JSON benchmark report",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image file does not exist: {args.image}")
    if any(c < 1 for c in args.batch_counts):
        parser.error("--batch-counts entries must be >= 1")
    if args.iterations < 1 or args.warmup < 0 or args.chunk_size < 1:
        parser.error("iterations, warmup, and chunk-size must be positive")

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Could not decode image: {args.image}")

    cv2.setNumThreads(1)
    detector = EmotionDetector()

    # Pre-verify face presence in base image
    initial_check = detector.analyze(image)
    detected_faces_per_img = len(initial_check)

    comparisons = []
    for count in args.batch_counts:
        batch_images = [image.copy() for _ in range(count)]
        seq_stats = benchmark_run(
            detector, batch_images, args.warmup, args.iterations, mode="sequential"
        )
        batch_stats = benchmark_run(
            detector,
            batch_images,
            args.warmup,
            args.iterations,
            mode="batched",
            batch_size=args.chunk_size,
        )

        seq_med = seq_stats["latency_ms"]["median"]  # type: ignore[index]
        batch_med = batch_stats["latency_ms"]["median"]  # type: ignore[index]
        speedup = round(seq_med / batch_med, 2) if batch_med > 0 else 1.0

        comparisons.append(
            {
                "batch_size": count,
                "total_faces_per_batch": count * detected_faces_per_img,
                "sequential": seq_stats["latency_ms"],
                "batched": batch_stats["latency_ms"],
                "speedup_ratio_median": speedup,
            }
        )

    report = {
        "benchmark": "ERML Sequential vs Batched Inference Latency",
        "image": str(args.image.resolve()),
        "faces_detected_in_base_image": detected_faces_per_img,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "not reported by platform",
            "logical_cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "onnxruntime": ort.__version__,
            "execution_provider": "CPUExecutionProvider",
        },
        "methodology": {
            "timed_region": "Sequential list loop vs detector.analyze_batch()",
            "opencv_threads": 1,
            "warmup_iterations": args.warmup,
            "timed_iterations": args.iterations,
            "onnx_chunk_size": args.chunk_size,
        },
        "results": comparisons,
    }

    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved benchmark report to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
