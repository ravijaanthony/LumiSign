# Start with a Python base image
FROM python:3.10

# Install Node.js (needed to build your React UI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Set the working directory
WORKDIR /app

# Copy your repository files into the Docker container
COPY . .

# 1. Build the React frontend
WORKDIR /app/ui
RUN npm install
RUN npm run build

# 2. Install Python dependencies
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
# Ensure uvicorn and python-multipart are installed for FastAPI
RUN pip install uvicorn python-multipart

# Expose the default Hugging Face port
EXPOSE 7860

# Command to run your FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]