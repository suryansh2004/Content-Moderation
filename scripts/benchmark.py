from __future__ import annotations

import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

import httpx

from moderation_api.model import load_model

SAMPLE_TEXTS = [
    "Thanks for the thoughtful explanation.",
    "You are an idiot and nobody wants you here.",
    "I disagree with the policy, but I understand your point.",
    "I will find you and make you regret this.",
    "This is obscene trash.",
    "Great work on the release today.",
    "People like you should disappear.",
    "Could you clarify the deployment steps?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--model-dir", default="JungleLee/bert-toxic-comment-classification")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batches = [make_batch(args.batch_size, i) for i in range(args.requests)]
    started = perf_counter()

    if args.api_url:
        latencies, flagged = benchmark_api(args, batches)
    else:
        latencies, flagged = benchmark_local(args, batches)

    wall_time = perf_counter() - started
    print_report(latencies, flagged, args.requests, args.batch_size, wall_time)


def benchmark_local(args: argparse.Namespace, batches: list[list[str]]) -> tuple[list[float], int]:
    model = load_model(args.model_dir, args.backend, args.max_length)
    latencies = []
    flagged = 0
    for batch in batches:
        results, latency_ms = model.predict(batch, threshold=args.threshold)
        latencies.append(latency_ms)
        flagged += sum(1 for item in results if item["flagged"])
    return latencies, flagged


def benchmark_api(args: argparse.Namespace, batches: list[list[str]]) -> tuple[list[float], int]:
    latencies = []
    flagged = 0

    def call(batch: list[str]) -> tuple[float, int]:
        started = perf_counter()
        response = httpx.post(
            args.api_url,
            json={"texts": batch, "threshold": args.threshold},
            timeout=30,
        )
        response.raise_for_status()
        latency_ms = (perf_counter() - started) * 1000.0
        payload = response.json()
        return latency_ms, sum(1 for item in payload["results"] if item["flagged"])

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(call, batch) for batch in batches]
        for future in as_completed(futures):
            latency_ms, flagged_count = future.result()
            latencies.append(latency_ms)
            flagged += flagged_count
    return latencies, flagged


def make_batch(batch_size: int, offset: int) -> list[str]:
    return [SAMPLE_TEXTS[(offset + i) % len(SAMPLE_TEXTS)] for i in range(batch_size)]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((pct / 100.0) * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


def print_report(
    latencies: list[float],
    flagged: int,
    requests: int,
    batch_size: int,
    wall_time: float,
) -> None:
    total_items = requests * batch_size
    print(f"Requests: {requests}")
    print(f"Batch size: {batch_size}")
    print(f"Items: {total_items}")
    print(f"Flagged items: {flagged}")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Throughput: {total_items / wall_time:.2f} items/s")
    print(f"Mean latency: {statistics.mean(latencies):.2f} ms")
    print(f"p50 latency: {percentile(latencies, 50):.2f} ms")
    print(f"p90 latency: {percentile(latencies, 90):.2f} ms")
    print(f"p95 latency: {percentile(latencies, 95):.2f} ms")
    print(f"p99 latency: {percentile(latencies, 99):.2f} ms")


if __name__ == "__main__":
    main()
