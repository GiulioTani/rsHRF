# Use the official Python 3.11 image to match the development environment
FROM python:3.11

# Disable interactive prompts during the build process
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies required for C-extensions and BIDS processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    build-essential \
                    libtool \
                    autoconf \
                    pkg-config \
                    libgsl-dev \
                    ca-certificates \
                    xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory for the application
WORKDIR /rsHRF

# Copy requirements first to take advantage of Docker's layer caching
# This prevents re-installing all libraries if only source code changes
COPY requirements.txt /rsHRF/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY . /rsHRF/

# Install the rsHRF package in editable mode
RUN pip install --no-cache-dir -e .

# Set the default command to run the rsHRF CLI
ENTRYPOINT ["rsHRF"]