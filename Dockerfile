# Use official Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (to leverage Docker cache)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Install Flask and pandas explicitly
RUN pip install Flask==3.0.0 pandas==2.1.4

# Create necessary directories
RUN mkdir -p /app/Data /app/preprocessed /app/models /app/uploads /app/results /app/templates

# Expose Flask default port
EXPOSE 5000

# Default command to run Flask app
CMD ["python", "app.py"]