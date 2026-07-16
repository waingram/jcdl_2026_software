# Dockerfile to create a container image using NVIDIA PyTorch as the
# base, setting up UV and a custom user environment for multiple
# developers.

# Author: Bill Ingram <waingram@vt.edu>
# Date: Fri Jan 30 11:57:18 AM EST 2026

# NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:26.01-py3

LABEL maintainer="Bill Ingram <waingram@vt.edu>"
ENV DEBIAN_FRONTEND=noninteractive  

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME

# Ensure required arguments are set
RUN if [ -z "$USER_ID" ] || [ -z "$GROUP_ID" ] || [ -z "$USER_NAME" ]; then \
        echo "ERROR: USER_ID, GROUP_ID, and USER_NAME must be provided!" >&2; exit 1; \
    fi

# -------------------------------
# 1. Install system dependencies as root
# -------------------------------
RUN apt-get update && \
    apt-get install -y sudo curl git && \
    rm -rf /var/lib/apt/lists/*

# -------------------------------
# 2. Install UV 
# -------------------------------
# This copies the compiled binary directly. It's cleaner than curling a script.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# -------------------------------
# 3. Configure Environment Variables
# -------------------------------
# Setting VIRTUAL_ENV tells 'uv' (and python) exactly where to look.
# We no longer need to manually 'source activate' in every step.
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# -------------------------------
# 4. Create the Virtual Environment
# -------------------------------
# CRITICAL: We use --system-site-packages so we inherit PyTorch/CUDA 
# from the NVIDIA base image.
RUN uv venv $VIRTUAL_ENV --system-site-packages

# -------------------------------
# 5. Create a non-root developer user
# -------------------------------
RUN echo ${GROUP_ID} && \
    addgroup --gid $GROUP_ID $USER_NAME && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME --force-badname && \
    echo $USER_NAME:$USER_NAME|chpasswd && \
    adduser ${USER_NAME} sudo && \
    echo ${USER_NAME} 'ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    mkdir -p /opt/app /workspace /home/$USER_NAME/.config

# -------------------------------
# 6. Install Dependencies
# -------------------------------
WORKDIR /workspace

# Copy the complete paper software package, then install from the lockfile.
COPY . .
RUN uv sync --frozen

# -------------------------------
# 7. Register Jupyter kernel
# -------------------------------
RUN python -m ipykernel install --sys-prefix --name uv-env --display-name "Python (UV)"

# -------------------------------
# 8. Set ownership for the developer user
# -------------------------------
# We chown the venv so the user can pip install more things if they want.
RUN chown -R $USER_NAME:$GROUP_ID $VIRTUAL_ENV /workspace /home/$USER_NAME/.config

# -------------------------------
# 9. Switch to non-root user
# -------------------------------
USER $USER_NAME
ENV PYTHONPATH=/workspace/src
WORKDIR /workspace
