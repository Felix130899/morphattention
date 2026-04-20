# 1. Use a stable PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# 2. Prevent Python from writing .pyc files and keep stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory
WORKDIR /workspace

# 4. Install system-level dependencies for OpenCV and Git
# We do this in one step and clean up to keep the image small
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install Python libraries
# We do this before copying the rest of the code to use Docker's cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Create a non-root user for better compatibility with host file permissions
# Replace '1000' with your actual UID/GID if they differ (1000 is default for the first Ubuntu user)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} vitgroup && \
    useradd -l -u ${USER_ID} -g vitgroup -m vituser && \
    chown -R vituser:vitgroup /workspace

USER vituser

# 7. Default command
CMD ["/bin/bash"]
