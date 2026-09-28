FROM python:3.11-slim

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install build tools and setuptools with pkg_resources (<70.0.0)
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir "setuptools<70.0.0"

# Copy requirements and install dependencies with setuptools available
COPY requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy application code
COPY . .

# Expose port and run uvicorn
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
