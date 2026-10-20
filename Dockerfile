# Образ UX-рецензента. Собирается GitHub Actions → ghcr.io/johnsmith-sg/ux-reviewer:latest.
# Ключа API в образе нет — на сервере он монтируется файлом (PROXYAPI_KEY_FILE).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py openai_module.py app.py ./
COPY static/ static/

RUN useradd --system --uid 10001 ux && mkdir /data && chown ux /data
USER ux
ARG GIT_SHA=local
ENV UX_DATA_DIR=/data \
    PYTHONUNBUFFERED=1 \
    UX_VERSION=$GIT_SHA
VOLUME /data
EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
