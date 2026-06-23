FROM python:3.11-slim

WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY rag_benchmark.db .

# Expose port (commonly 8080 or dynamic $PORT)
EXPOSE 8080

# Command to run application, supporting dynamic port bindings
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
