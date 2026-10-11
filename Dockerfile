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

# 5. Copy requirements and install Python libraries (pinned; constraints.txt
# pins their sub-dependencies). We do this before copying the rest of the code
# to use Docker's cache
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# 5b. Point the Hugging Face cache at the bind-mounted data/ folder instead
# of baking model weights into the image. data/ already lives on the host
# (see docker-compose.yml volume mount) and is gitignored, so weights persist
# across container rebuilds without bloating the image or the git repo.
# Run scripts/download_sam.py once after first build to populate it.
ENV HF_HOME=/workspace/data/model_cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV PYTHONPATH=/workspace/src

# 6. Create a non-root user with your host UID/GID, so files written in the
# container belong to you on the host. docker-compose.yml passes them from .env
# (default 1000, the first Ubuntu user). A GID that already exists in the image
# (e.g. 20 on macOS) is reused instead of created.
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN (getent group ${GROUP_ID} || groupadd -g ${GROUP_ID} vitgroup) && \
    useradd -l -u ${USER_ID} -g ${GROUP_ID} -m vituser && \
    chown -R vituser:${GROUP_ID} /workspace

USER vituser

# 7. Default command
CMD ["/bin/bash"]
