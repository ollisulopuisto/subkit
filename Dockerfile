FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY subkit ./subkit
RUN uv pip install --system --no-cache ".[asr-sherpa]"

ENTRYPOINT ["subkit"]
