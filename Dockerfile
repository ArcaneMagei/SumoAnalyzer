# Fixed Dockerfile - Resolves apt-get exit code 100

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Fix apt-get issues by cleaning and updating properly
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        curl \
        wget \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads and results
RUN mkdir -p /app/uploads /app/results /app/data

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
