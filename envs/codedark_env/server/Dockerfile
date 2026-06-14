FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY codedark/ ./codedark/
COPY data/ ./data/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CODEDARK_DATA_DIR=/app/data
ENV CODEDARK_TASKS_PATH=/app/data/tasks/final_25_tasks.jsonl
ENV HOST=0.0.0.0
ENV PORT=7860

# Expose HuggingFace Spaces port
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run server
CMD ["python", "-m", "uvicorn", "codedark.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
