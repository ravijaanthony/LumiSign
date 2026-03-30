# Start with a Python base image
FROM python:3.9
# Create a non-root user with ID 1000 (Required by Hugging Face)
RUN useradd -m -u 1000 user

# Install Node.js (for UI build) and OpenCV runtime system libs.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends \
        nodejs \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your repository files and immediately give ownership to the new user
COPY --chown=user:user . .

# Switch away from the root user to the Hugging Face required user
USER user

# Ensure installed Python packages are added to the system path
ENV PATH="/home/user/.local/bin:${PATH}"

# 1. Build the React frontend
WORKDIR /app/ui
RUN npm install
RUN npm run build

# 2. Install Python dependencies
WORKDIR /app
# Upgrade pip first, then install requirements
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn python-multipart

# Expose the default Hugging Face port
EXPOSE 7860

# Runtime model configuration (override these from HF Space Variables if needed)
ENV MODEL_DATASET=Dataset
ENV MODEL_TYPE=transformer
ENV MODEL_TRANSFORMER_SIZE=large
ENV MODEL_MAX_FRAME_LEN=169
ENV MODEL_CHECKPOINT=/app/transformer_large.pth

# Command to run your FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
