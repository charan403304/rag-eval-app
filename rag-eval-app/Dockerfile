FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the offline index at image build time so the container starts fast.
# Override EMBEDDING_PROVIDER at runtime (docker run -e EMBEDDING_PROVIDER=voyage ...)
# and re-run scripts/build_index.py if you want a real-embedding index instead.
RUN python scripts/build_index.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
