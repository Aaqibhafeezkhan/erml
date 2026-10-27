"""Reproducible CPU-only end-to-end inference latency benchmark.

Usage:
    python scripts/benchmark_cpu.py path/to/image.jpg
    python scripts/benchmark_cpu.py path/to/image.jpg --iterations 50 --warmup 5
    python scripts/benchmark_cpu.py path/to/image.jpg --sizes 320x240 640x480 1280x720 --json results.json

Detector initialization/model download is excluded. Timings include face detection and
per-face expression inference for each resized image. Run on an otherwise idle machine.
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


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            "size must be WIDTHxHEIGHT, e.g. 640x480"
        ) from exc
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def percentile(values: list[float], percent: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percent))


def benchmark_size(
    detector: EmotionDetector,
    image: np.ndarray,
    size: tuple[int, int],
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    width, height = size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    for _ in range(warmup):
        detector.analyze(resized)

    durations_ms = []
    face_count = 0
    for _ in range(iterations):
        started = time.perf_counter()
        results = detector.analyze(resized)
        durations_ms.append((time.perf_counter() - started) * 1000)
        face_count = len(results)

    return {
        "width": width,
        "height": height,
        "iterations": iterations,
        "warmup_iterations": warmup,
        "faces_detected_last_iteration": face_count,
        "latency_ms": {
            "min": round(min(durations_ms), 3),
            "median": round(statistics.median(durations_ms), 3),
            "p95": round(percentile(durations_ms, 95), 3),
            "max": round(max(durations_ms), 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="path to a representative image")
    parser.add_argument(
        "--iterations", type=int, default=30, help="timed runs per size (default: 30)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="untimed runs per size (default: 5)"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=parse_size,
        default=[(320, 240), (640, 480), (1280, 720)],
        metavar="WIDTHxHEIGHT",
        help="target image sizes (default: 320x240 640x480 1280x720)",
    )
    parser.add_argument(
        "--json", type=Path, help="also write the complete results to this JSON file"
    )
    args = parser.parse_args()

    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup cannot be negative")
    if not args.image.is_file():
        parser.error(f"image file does not exist: {args.image}")

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"could not decode image: {args.image}")

    cv2.setNumThreads(1)
    detector = EmotionDetector()
    results = [
        benchmark_size(detector, image, size, args.warmup, args.iterations)
        for size in args.sizes
    ]

    report = {
        "benchmark": "ERML CPU end-to-end inference latency",
        "image": str(args.image.resolve()),
        "source_image_dimensions": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
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
            "timed_region": "EmotionDetector.analyze() only; initialization/download excluded",
            "includes": "face detection and expression inference for detected faces",
            "resize_interpolation": "cv2.INTER_AREA",
            "opencv_threads": 1,
            "warmup_iterations_per_size": args.warmup,
            "timed_iterations_per_size": args.iterations,
            "statistics": ["min", "median", "p95", "max"],
        },
        "results": results,
    }

    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved benchmark report to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
