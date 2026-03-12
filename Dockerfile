FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        firefox-esr \
        tar \
        wget \
    && rm -rf /var/lib/apt/lists/*

ARG GECKODRIVER_VERSION=0.36.0

RUN wget -q "https://github.com/mozilla/geckodriver/releases/download/v${GECKODRIVER_VERSION}/geckodriver-v${GECKODRIVER_VERSION}-linux64.tar.gz" \
    && tar -xzf "geckodriver-v${GECKODRIVER_VERSION}-linux64.tar.gz" -C /usr/local/bin \
    && chmod +x /usr/local/bin/geckodriver \
    && rm "geckodriver-v${GECKODRIVER_VERSION}-linux64.tar.gz"

COPY pyproject.toml README.md ./
COPY fetch_content.py main.py query_expansion.py search_retrieval.py url_reranking.py ./

RUN pip install --no-cache-dir setuptools wheel \
    && pip install --no-cache-dir --no-build-isolation .

RUN python -m playwright install --with-deps chromium

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
