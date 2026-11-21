# Production image for the Streamlit app.
# Expects trained artifacts in models/ (run `uv run python -m salary.train` first).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" PORT=8501

# OpenMP runtime for XGBoost / LightGBM, curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

COPY .streamlit ./.streamlit
COPY app ./app
COPY data/raw ./data/raw
COPY models ./models
COPY reports ./reports

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app
ENV SALARY_ROOT=/app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS "http://localhost:${PORT}/_stcore/health" || exit 1
CMD ["sh", "-c", "streamlit run app/Home.py --server.port=${PORT} --server.address=0.0.0.0"]
