FROM python:3.12-slim

WORKDIR /app

# System deps needed by some Python packages (networkx, pandas build deps)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY diagnotix/requirements.txt diagnotix/requirements.txt
COPY knowledge-graphs/requirements.txt knowledge-graphs/requirements.txt
RUN pip install --no-cache-dir \
    -r diagnotix/requirements.txt \
    -r knowledge-graphs/requirements.txt

# Copy application code
COPY diagnotix/ diagnotix/
COPY knowledge-graphs/ knowledge-graphs/

# backend imports from knowledge-graphs/ at runtime (build_kg, audit_guidelines, etc.)
ENV PYTHONPATH=/app/diagnotix:/app/knowledge-graphs

WORKDIR /app/diagnotix
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
