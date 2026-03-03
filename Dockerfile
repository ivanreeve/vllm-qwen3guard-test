FROM python:3.13-alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Keep runtime image lean: install only evaluator runtime dependencies.
COPY requirements.docker.txt /tmp/requirements.docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements.docker.txt

# Create non-root user.
RUN adduser -D appuser

COPY detect_pii.py /app/detect_pii.py
COPY data /app/data

USER appuser

ENTRYPOINT ["python", "detect_pii.py"]
