# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel

# Install pandas and Flask explicitly to avoid source build issues
RUN pip install pandas==2.1.4 Flask==3.0.0
RUN pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Train the model during build
RUN python model_trainer.py

# Expose Flask port
EXPOSE 5000

# Default command: do nothing (use Makefile to run app)
CMD ["tail", "-f", "/dev/null"]
