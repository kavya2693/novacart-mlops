# Slice 2 — Kubernetes serving + horizontal scaling

Deploys the four real NovaCart services (churn prediction, feature, forecast,
graph-RAG) to a local Kubernetes cluster (`kind`) via a Helm chart, and measures
what horizontal scaling actually does to throughput and tail latency.

## What runs
- **Image** `novacart:baked` — the real training pipeline baked at build time so
  every pod boots from identical model + feature artifacts. (Prod would split a
  train Job from serving Deployments over the registry; the baked image is a
  deliberate, stated trade for a self-contained demo.)
- **Helm chart** `deploy/helm/novacart` — 4 Deployments, 4 Services (in-cluster
  DNS wiring via `NOVACART_FEATURE_SERVICE_URL`), health probes, and an HPA on the
  prediction service (1→5 replicas at 70% CPU). `helm lint` clean; renders 9 objects.
- **Cluster** single-node `kind` inside a colima VM with **2 vCPUs** — the binding
  constraint below.

## The scaling experiment (in-cluster, load-balanced Service, 1500 req, conc 16–24)

| Configuration | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| prediction=1, feature=1 | 73.0 req/s | 214 ms | 303 ms | 350 ms |
| **prediction=3**, feature=1 | 73.7 req/s | 212 ms | 298 ms | 342 ms |
| prediction=3, **feature=3** | **157.0 req/s** | 138 ms | 302 ms | 364 ms |

## The finding (this is the point)
Scaling the prediction service from 1→3 replicas gave **zero** throughput gain.
That is not a bug — every `/predict` makes a synchronous call to the
**feature-service**, so a single feature-service replica caps the whole path.
Scaling the *actual* constraint (feature-service to 3) lifted throughput **2.15×**
(73→157 req/s) and cut p50 by 35%, even on a 2-vCPU host.

Lesson: add replicas to the bottleneck in the critical path, not to the service
whose name is on the endpoint. Found by benchmarking, not guessing.

## Honest limitations
- 2-vCPU colima VM caps absolute numbers; the ~2× ceiling is the host, not the design.
- Load generator runs in-cluster (a neutral pod) hitting the Service ClusterIP so
  the numbers include real load-balancing; absolute values shift on real hardware,
  relative comparisons hold.
- Baked-artifact image is a demo trade-off, stated above.

## Verified
- All 4 pods Ready; in-cluster prediction returns the same score as local
  (customer 10 → 0.8599) → serving parity.
- metrics-server installed (`--kubelet-insecure-tls` for kind); HPA reads live CPU
  (`4%/70%`) and will autoscale prediction 1→5 under load.
