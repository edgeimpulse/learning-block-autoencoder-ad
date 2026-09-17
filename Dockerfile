# Simple Ubuntu 24.04 base image with Python3.12 and CUDA setup already (for GPU training)
FROM public.ecr.aws/g7a8t7v6/ei-custom-ml-block-base:v1.95.5

# Install uv and Python 3.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv python install 3.12
ENV UV_LINK_MODE=copy

# Copy Python project files in and install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Copy the rest of your training scripts in
COPY . ./

# And tell us where to run the pipeline
ENTRYPOINT ["uv", "run", "python3", "-u", "train.py"]
