# Production image for all NovaCart services (one image, many entrypoints).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt

COPY novacart ./novacart
COPY scripts ./scripts

# Default: run the batch pipeline. Compose overrides `command` per service.
CMD ["python", "-m", "novacart.data_engine.generate"]
