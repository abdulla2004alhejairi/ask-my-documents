FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app/ app/
COPY data/ data/

# Bake the embedding model cache and the FAISS index into the image.
RUN python -m app.ingest

RUN useradd --create-home appuser && chown -R appuser /srv /opt/hf
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
