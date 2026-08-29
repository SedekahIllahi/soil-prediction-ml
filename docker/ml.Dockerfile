FROM python:3.11-slim

WORKDIR /app

# Install build dependencies (needed for some ML packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create volume mount points for the application and data
RUN mkdir -p /app/storage/datasets
RUN mkdir -p /app/experiments/baseline

# Ensure Python can find the modules
ENV PYTHONPATH=/app

# The default command will run the baseline experiment, but can be overridden
CMD ["python", "-m", "ml.experiments.run_baseline"]
