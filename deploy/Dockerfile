# Serving image with model + feature artifacts baked in.
#
# For a local kind demo the simplest honest approach is to run the training
# pipeline at build time so every pod boots from the SAME artifacts. In real
# production you separate train (a Job) from serve (Deployments) and share the
# registry over an artifact store — the Registry API here is already shaped for
# that. This image trades that separation for a self-contained, reproducible
# demo, and that trade-off is stated, not hidden.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# LightGBM needs the OpenMP runtime (libgomp), absent from the slim base.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt

COPY novacart ./novacart
COPY scripts ./scripts

# Bake artifacts: data -> features -> churn model -> forecast -> graph.
RUN python -m novacart.data_engine.generate \
 && python -m novacart.feature_pipeline.build_features \
 && python -m novacart.training.train \
 && python -m novacart.forecasting.train \
 && python -m novacart.graph.build_graph \
 && python -m novacart.graph.clv

EXPOSE 8101 8102 8103 8104

# Compose/Helm override `command` per service.
CMD ["python", "-m", "novacart.serving.feature_service"]
