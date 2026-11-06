# syntax=docker/dockerfile:1
#
# Container image for the OULAD at-risk prediction pipeline.
#
# Stages
#   base     Python and environment settings shared by every stage
#   deps     pinned runtime dependencies (a cached layer, rebuilt only when the lock changes)
#   test     deps + PyTorch (CPU) + test tools + source code; CI runs lint and tests in it
#   builder  builds the wheel of the package
#   runtime  (default) deps + the wheel, runs as a non-root user; this is the shipped image
#
# Examples
#   docker build -t oulad-analytics .                     # runtime image
#   docker build --target test -t oulad-analytics-test .  # test image
#   docker run --rm oulad-analytics                       # demo on synthetic data
#   docker run --rm -p 8000:8000 oulad-analytics serve    # demo + web page at http://localhost:8000
#   docker run --rm -v C:\data\OULAD:/data:ro -v %cd%\results:/results \
#       oulad-analytics train --data-dir /data --out-dir /results

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------- base
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg
WORKDIR /app

# ---------------------------------------------------------------- deps
FROM base AS deps
COPY requirements/runtime.txt /tmp/requirements/runtime.txt
RUN pip install -r /tmp/requirements/runtime.txt

# ---------------------------------------------------------------- test
FROM deps AS test
# CPU-only PyTorch is enough for the LSTM tests and is ~10x smaller than the CUDA build.
ARG TORCH_VERSION=2.15.0
RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
COPY requirements/dev.txt /tmp/requirements/dev.txt
RUN pip install -r /tmp/requirements/dev.txt
COPY . .
RUN pip install --no-deps -e .
# Default command = the unit-test run used by Jenkins; reports are copied out with `docker cp`.
CMD ["sh", "-c", "mkdir -p reports && pytest --cov --cov-report=term --cov-report=xml:reports/coverage.xml --cov-report=html:reports/htmlcov --junitxml=reports/junit.xml"]

# ---------------------------------------------------------------- builder
FROM base AS builder
RUN pip install "build==1.6.1"
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

# ---------------------------------------------------------------- runtime
FROM deps AS runtime
ARG VERSION=dev
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="oulad-analytics" \
      org.opencontainers.image.description="Early prediction of at-risk students from OULAD learning-analytics data" \
      org.opencontainers.image.source="https://github.com/Al0nesh/oulad-student-analytics" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-deps /tmp/*.whl && rm /tmp/*.whl && pip check
COPY docker/entrypoint.py /app/entrypoint.py
RUN useradd --create-home --uid 10001 app && mkdir -p /results && chown app /results
USER 10001
ENV APP_VERSION=${VERSION} \
    APP_COMMIT=${VCS_REF} \
    RESULTS_DIR=/results \
    PORT=8000
EXPOSE 8000
ENTRYPOINT ["python", "/app/entrypoint.py"]
CMD ["demo", "--out-dir", "/results"]
