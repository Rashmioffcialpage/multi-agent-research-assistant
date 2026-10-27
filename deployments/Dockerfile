# Build from the repo root: docker build -f deployments/Dockerfile -t research-assistant .
#
# Pulled via mirror.gcr.io (Google's public, unauthenticated Docker Hub
# mirror — the same one GKE nodes use by default) rather than docker.io
# directly, for resilience against Docker Hub rate limits/outages.
FROM mirror.gcr.io/library/python:3.11-slim

WORKDIR /app

COPY pyproject.toml /app/
COPY research_assistant /app/research_assistant
COPY data /app/data

RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "research_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
