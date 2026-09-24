FROM node:22-slim AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY backend ./backend
COPY src ./src
COPY models ./models
COPY artifacts ./artifacts
COPY data/sgcc_demo.csv.gz ./data/sgcc_demo.csv.gz
COPY config.yaml ./

RUN useradd --create-home --uid 10001 app && mkdir -p artifacts/state && chown -R app:app artifacts
USER app
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]