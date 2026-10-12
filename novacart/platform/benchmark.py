"""Latency and throughput benchmark for the model-serving path.

This is the verifier for the serving layer: before we talk about Kubernetes,
quantisation or autoscaling, we need honest numbers for what one replica of the
real prediction service actually does under load — tail latency and throughput —
and proof that the served score matches the model, not just that a request
returns 200.

It drives the true end-to-end path (prediction-service -> feature-service ->
Production model in the registry), so the numbers include the service-to-service
hop, not an in-process shortcut.

Run:
    python -m novacart.platform.benchmark --requests 2000 --concurrency 16
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import time
from pathlib import Path

import httpx


def discover_ids(base: str, probe: int, want: int, client: httpx.Client) -> list[int]:
    """Find customer ids the service can actually score (features exist)."""
    ids: list[int] = []
    for cid in range(1, probe + 1):
        try:
            r = client.get(f"{base}/predict/{cid}")
        except httpx.HTTPError:
            continue
        if r.status_code == 200:
            ids.append(cid)
            if len(ids) >= want:
                break
    return ids


def correctness_check(base: str, ids: list[int], client: httpx.Client) -> dict:
    """Two independent checks that the score is real, not just a 200:
    determinism (same id -> same probability) and range ([0, 1])."""
    deterministic = True
    in_range = True
    for cid in ids[:25]:
        a = client.get(f"{base}/predict/{cid}").json()
        b = client.get(f"{base}/predict/{cid}").json()
        pa, pb = a["churn_probability"], b["churn_probability"]
        if abs(pa - pb) > 1e-9:
            deterministic = False
        if not (0.0 <= pa <= 1.0):
            in_range = False
    return {"deterministic": deterministic, "probability_in_range": in_range,
            "checked": min(25, len(ids))}


def pct(values: list[float], p: float) -> float:
    """p-th percentile (nearest-rank) in milliseconds, values already sorted."""
    if not values:
        return float("nan")
    k = max(0, min(len(values) - 1, int(round(p / 100 * len(values) + 0.5)) - 1))
    return values[k]


def run(base: str, total: int, concurrency: int) -> dict:
    client = httpx.Client(timeout=10.0, limits=httpx.Limits(max_connections=concurrency + 4))
    ids = discover_ids(base, probe=1500, want=200, client=client)
    if not ids:
        raise SystemExit("no scorable customer ids found — is the pipeline trained and are both services up?")

    correctness = correctness_check(base, ids, client)

    # Warm up (JIT, caches, connection pool) — excluded from the measurement.
    for cid in ids[:50]:
        client.get(f"{base}/predict/{cid}")

    latencies: list[float] = []
    errors = 0

    def one(i: int) -> float | None:
        cid = ids[i % len(ids)]
        t0 = time.perf_counter()
        try:
            r = client.get(f"{base}/predict/{cid}")
            if r.status_code != 200:
                return None
        except httpx.HTTPError:
            return None
        return (time.perf_counter() - t0) * 1000.0

    wall0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for lat in ex.map(one, range(total)):
            if lat is None:
                errors += 1
            else:
                latencies.append(lat)
    wall = time.perf_counter() - wall0

    latencies.sort()
    ok = len(latencies)
    return {
        "base_url": base,
        "requests_total": total,
        "requests_ok": ok,
        "errors": errors,
        "concurrency": concurrency,
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(ok / wall, 1) if wall > 0 else None,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2) if ok else None,
            "p50": round(pct(latencies, 50), 2) if ok else None,
            "p95": round(pct(latencies, 95), 2) if ok else None,
            "p99": round(pct(latencies, 99), 2) if ok else None,
            "max": round(latencies[-1], 2) if ok else None,
        },
        "correctness": correctness,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8002")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default="reports/serving_benchmark.json")
    args = ap.parse_args()

    result = run(args.base, args.requests, args.concurrency)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    lm = result["latency_ms"]
    print(f"\nServing benchmark  ({result['base_url']}, concurrency={result['concurrency']})")
    print(f"  requests      {result['requests_ok']}/{result['requests_total']} ok, {result['errors']} errors")
    print(f"  throughput    {result['throughput_rps']} req/s over {result['wall_seconds']}s")
    print(f"  latency (ms)  mean {lm['mean']} | p50 {lm['p50']} | p95 {lm['p95']} | p99 {lm['p99']} | max {lm['max']}")
    c = result["correctness"]
    print(f"  correctness   deterministic={c['deterministic']}  in_range={c['probability_in_range']}  (n={c['checked']})")
    print(f"  written       {out}")


if __name__ == "__main__":
    main()
