#!/usr/bin/env bash
# Source from the repository root; no persistent shell configuration is changed.
export KILLFIELD_ROOT="$(pwd -P)"
export KILLFIELD_RUNTIME="$(dirname "$KILLFIELD_ROOT")/.killfield-runtime"
export UV_CACHE_DIR="$KILLFIELD_RUNTIME/uv-cache"
export UV_PYTHON_INSTALL_DIR="$KILLFIELD_RUNTIME/python"
export UV_PROJECT_ENVIRONMENT="$KILLFIELD_RUNTIME/venv"
export CARGO_HOME="$KILLFIELD_RUNTIME/cargo"
export RUSTUP_HOME="$KILLFIELD_RUNTIME/rustup"
export RUSTUP_TOOLCHAIN=1.88.0
export CARGO_TARGET_DIR="$KILLFIELD_RUNTIME/cargo-target"
export TMPDIR="$KILLFIELD_RUNTIME/tmp"
export XDG_CACHE_HOME="$KILLFIELD_RUNTIME/xdg-cache"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
