# Base Image: Python 3.12 Slim
FROM python:3.12-slim

# Prevents Python from writing .pyc files and enables unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

# Working directory inside container
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project source code, models, data, and static assets
COPY . .

# Ensure data and model artifacts are generated and validated
RUN python generate_data.py && python train.py

# Expose port (default 10000)
EXPOSE 10000

# Start Uvicorn server respecting the $PORT environment variable
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
