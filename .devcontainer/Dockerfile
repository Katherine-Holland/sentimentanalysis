FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Install Python and pip
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python3 python3-pip python3-venv \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*
