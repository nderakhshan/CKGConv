#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# CKGConv one-command installer
#
# Intended platform:
#   - Ubuntu/Debian Linux
#   - x86_64
#   - NVIDIA GPU with working driver
#
# This script:
#   1. Installs Git, curl, wget, and build tools
#   2. Installs Miniconda if needed
#   3. Creates the ckgconv Python 3.9 environment
#   4. Installs the original CKGConv dependencies
#   5. Installs this repository
#   6. Verifies PyTorch, CUDA, and PyG
# ============================================================

ENV_NAME="ckgconv"
REPO_URL="https://github.com/nderakhshan/CKGConv.git"
REPO_DIR="${CKGCONV_DIR:-$HOME/CKGConv}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"

log() {
    printf '\n\033[1;34m[CKGConv]\033[0m %s\n' "$1"
}

fail() {
    printf '\n\033[1;31m[ERROR]\033[0m %s\n' "$1" >&2
    exit 1
}

# Do not run the whole script as root.
if [[ "${EUID}" -eq 0 ]]; then
    fail "Run this script as a normal user, not with sudo."
fi

# Detect sudo availability.
if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi

# ------------------------------------------------------------
# 1. Check the GPU driver
# ------------------------------------------------------------

log "Checking NVIDIA GPU and driver"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi was not found."
    echo "The provider must supply an NVIDIA driver before GPU training can work."
fi

# ------------------------------------------------------------
# 2. Install operating-system utilities
# ------------------------------------------------------------

log "Installing operating-system utilities"

if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update
    $SUDO apt-get install -y \
        git \
        curl \
        wget \
        ca-certificates \
        build-essential \
        libgl1 \
        libglib2.0-0
else
    fail "This installer currently supports Ubuntu/Debian servers with apt-get."
fi

# ------------------------------------------------------------
# 3. Install Miniconda
# ------------------------------------------------------------

if [[ ! -x "$MINICONDA_DIR/bin/conda" ]]; then
    log "Installing Miniconda in $MINICONDA_DIR"

    ARCH="$(uname -m)"

    case "$ARCH" in
        x86_64)
            MINICONDA_FILE="Miniconda3-latest-Linux-x86_64.sh"
            ;;
        aarch64|arm64)
            MINICONDA_FILE="Miniconda3-latest-Linux-aarch64.sh"
            ;;
        *)
            fail "Unsupported CPU architecture: $ARCH"
            ;;
    esac

    INSTALLER="/tmp/$MINICONDA_FILE"

    curl -fsSL \
        "https://repo.anaconda.com/miniconda/$MINICONDA_FILE" \
        -o "$INSTALLER"

    bash "$INSTALLER" -b -p "$MINICONDA_DIR"
    rm -f "$INSTALLER"
else
    log "Miniconda is already installed"
fi

CONDA="$MINICONDA_DIR/bin/conda"

# Accept Anaconda channel terms where required by recent conda versions.
"$CONDA" tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true

"$CONDA" tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

# ------------------------------------------------------------
# 4. Clone or update the repository
# ------------------------------------------------------------

if [[ -d "$REPO_DIR/.git" ]]; then
    log "Updating repository in $REPO_DIR"

    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" pull --ff-only
else
    log "Cloning repository into $REPO_DIR"

    git clone "$REPO_URL" "$REPO_DIR"
fi

# ------------------------------------------------------------
# 5. Create the Conda environment
# ------------------------------------------------------------

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Conda environment '$ENV_NAME' already exists"
else
    log "Creating Conda environment '$ENV_NAME' with Python 3.9"

    "$CONDA" create -y \
        -n "$ENV_NAME" \
        python=3.9 \
        pip
fi

run_in_env() {
    "$CONDA" run --no-capture-output -n "$ENV_NAME" "$@"
}

# ------------------------------------------------------------
# 6. Install PyTorch 2.1.2 with CUDA 11.8
# ------------------------------------------------------------

log "Installing PyTorch 2.1.2 with CUDA 11.8"

run_in_env python -m pip install --upgrade \
    "pip<25" \
    wheel

run_in_env python -m pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --trusted-host download.pytorch.org

# ------------------------------------------------------------
# 7. Install PyTorch Geometric and compiled extensions
# ------------------------------------------------------------

log "Installing PyTorch Geometric"

run_in_env python -m pip install \
    torch_geometric==2.5.3 \
    --trusted-host data.pyg.org

run_in_env python -m pip install \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    --trusted-host data.pyg.org

# ------------------------------------------------------------
# 8. Install Conda packages
# ------------------------------------------------------------

log "Installing RDKit, OpenBabel, and fsspec"

"$CONDA" install -y \
    -n "$ENV_NAME" \
    -c conda-forge \
    openbabel \
    fsspec \
    rdkit

# ------------------------------------------------------------
# 9. Install the remaining Python dependencies
# ------------------------------------------------------------

log "Installing remaining CKGConv dependencies"

run_in_env python -m pip install \
    torchmetrics==0.9.1 \
    ogb \
    tensorboardX \
    yacs \
    opt_einsum \
    graphgym \
    pytorch-lightning \
    setuptools==59.5.0 \
    timm \
    einops \
    mlflow

# ------------------------------------------------------------
# 10. Install the local CKGConv repository
# ------------------------------------------------------------

log "Installing the CKGConv repository"

if [[ -f "$REPO_DIR/setup.py" || -f "$REPO_DIR/pyproject.toml" ]]; then
    run_in_env python -m pip install -e "$REPO_DIR"
else
    echo "No setup.py or pyproject.toml found; skipping editable installation."
fi

# ------------------------------------------------------------
# 11. Verification
# ------------------------------------------------------------

log "Verifying installation"

run_in_env python - <<'PY'
import sys

import torch
import torch_geometric

print("=" * 60)
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("PyTorch Geometric:", torch_geometric.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: PyTorch cannot access an NVIDIA GPU.")

modules = [
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "torch_spline_conv",
    "ogb",
    "rdkit",
    "mlflow",
    "yacs",
    "einops",
]

for module in modules:
    try:
        __import__(module)
        print(f"[OK] {module}")
    except Exception as exc:
        print(f"[FAILED] {module}: {exc}")
        raise

print("=" * 60)
print("CKGConv environment is ready.")
PY

# ------------------------------------------------------------
# 12. Final instructions
# ------------------------------------------------------------

log "Installation completed successfully"

echo
echo "Repository:"
echo "  $REPO_DIR"
echo
echo "To activate the environment:"
echo "  source \"$MINICONDA_DIR/etc/profile.d/conda.sh\""
echo "  conda activate \"$ENV_NAME\""
echo "  cd \"$REPO_DIR\""
echo
echo "Or run without activation:"
echo "  \"$CONDA\" run --no-capture-output -n \"$ENV_NAME\" python main.py --help"
echo
