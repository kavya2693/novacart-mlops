"""Tiny dependency-free Prometheus-style metrics.

Enough to expose /metrics with request counts, latency, and the live
prediction distribution — without pulling in prometheus_client. In real prod
you'd swap this for prometheus_client + a Grafana board; the endpoint contract
is identical.
"""
from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self._hist_sum: dict[str, float] = defaultdict(float)
        self._hist_count: dict[str, int] = defaultdict(int)

    def inc(self, name: str, labels: dict | None = None, value: float = 1.0) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._hist_sum[name] += value
            self._hist_count[name] += 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), val in sorted(self._counters.items()):
                lbl = ",".join(f'{k}="{v}"' for k, v in labels)
                lines.append(f"{name}{{{lbl}}} {val}" if lbl else f"{name} {val}")
            for name, total in sorted(self._hist_sum.items()):
                cnt = self._hist_count[name]
                lines.append(f"{name}_sum {total}")
                lines.append(f"{name}_count {cnt}")
                if cnt:
                    lines.append(f"{name}_avg {total / cnt}")
        return "\n".join(lines) + "\n"


metrics = Metrics()


def add_metrics_endpoint(app) -> None:
    """Mount a Prometheus-style GET /metrics on any FastAPI app (shared boilerplate)."""
    from fastapi.responses import PlainTextResponse

    @app.get("/metrics", response_class=PlainTextResponse)
    def _prometheus() -> str:  # noqa: D401 - route handler
        return metrics.render()
