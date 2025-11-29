# Use official Python 3.11 (no LibreSSL problem)
FROM python:3.11-slim

# Install system dependencies you often need
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip to avoid old dependencies conflicts
RUN pip install --upgrade pip setuptools wheel

# Install dependencies
RUN pip install -r requirements.txt

# Copy your project files
COPY . .

# Expose API port
EXPOSE 8080

# Run backend
CMD ["python", "main.py"]
