#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# CKGConv one-command installer
#
# Intended platform:
#   - Ubuntu/Debian Linux
#   - NVIDIA GPU with working driver
#
# Reproduction baseline:
#   - Python 3.9
#   - PyTorch 2.1.2 + CUDA 11.8
#   - PyTorch Geometric 2.5.3
#   - NumPy 1.26.4
#   - scikit-learn 1.5.2
#
# This script:
#   1. Installs OS utilities
#   2. Installs Miniforge from GitHub if Conda is missing
#   3. Clones/updates the CKGConv repository
#   4. Creates the ckgconv Python 3.9 environment
#   5. Installs CKGConv dependencies
#   6. Pins NumPy/scikit-learn to the versions verified to work
#   7. Verifies PyTorch, CUDA, PyG, and Torch<->NumPy conversion
#
# Optional environment variables:
#   CKGCONV_DIR=/custom/path
#   CKGCONV_CONDA_DIR=/custom/conda/path
#   CKGCONV_APT_LOCK_TIMEOUT=600
# ============================================================

ENV_NAME="ckgconv"
REPO_URL="https://github.com/nderakhshan/CKGConv.git"
REPO_DIR="${CKGCONV_DIR:-$HOME/CKGConv}"

# Keep ~/miniconda3 as the default path for backward compatibility
# with existing servers, but the distribution installed is Miniforge.
CONDA_DIR="${CKGCONV_CONDA_DIR:-${MINICONDA_DIR:-$HOME/miniconda3}}"
APT_LOCK_TIMEOUT="${CKGCONV_APT_LOCK_TIMEOUT:-600}"

log() {
    printf '\n\033[1;34m[CKGConv]\033[0m %s\n' "$1"
}

warn() {
    printf '\n\033[1;33m[WARNING]\033[0m %s\n' "$1" >&2
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
    warn "nvidia-smi was not found. The provider must supply an NVIDIA driver before GPU training can work."
fi

# ------------------------------------------------------------
# 2. Install operating-system utilities
# ------------------------------------------------------------

log "Installing operating-system utilities"

if command -v apt-get >/dev/null 2>&1; then
    # apt-get update can run while unattended-upgrades owns the dpkg lock,
    # but package installation cannot. DPkg::Lock::Timeout makes apt wait
    # instead of immediately failing on a fresh Ubuntu server.
    $SUDO apt-get update

    if ! $SUDO apt-get \
        -o "DPkg::Lock::Timeout=${APT_LOCK_TIMEOUT}" \
        install -y \
        git \
        curl \
        wget \
        ca-certificates \
        build-essential \
        libgl1 \
        libglib2.0-0; then

        fail "apt package installation failed. If unattended-upgrades is still running, wait for it to finish or stop it cleanly, then rerun this installer."
    fi
else
    fail "This installer currently supports Ubuntu/Debian servers with apt-get."
fi

# ------------------------------------------------------------
# 3. Install Miniforge (GitHub-hosted Conda distribution)
# ------------------------------------------------------------

if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
    log "Installing Miniforge in $CONDA_DIR"

    ARCH="$(uname -m)"

    case "$ARCH" in
        x86_64)
            MINIFORGE_FILE="Miniforge3-Linux-x86_64.sh"
            ;;
        aarch64|arm64)
            MINIFORGE_FILE="Miniforge3-Linux-aarch64.sh"
            ;;
        *)
            fail "Unsupported CPU architecture: $ARCH"
            ;;
    esac

    INSTALLER="/tmp/$MINIFORGE_FILE"
    MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE_FILE"

    curl -fL \
        --retry 5 \
        --retry-delay 3 \
        --retry-all-errors \
        "$MINIFORGE_URL" \
        -o "$INSTALLER"

    bash "$INSTALLER" -b -p "$CONDA_DIR"
    rm -f "$INSTALLER"
else
    log "Conda is already installed in $CONDA_DIR"
fi

CONDA="$CONDA_DIR/bin/conda"

# Miniforge uses conda-forge by default. No Anaconda ToS acceptance is needed.
"$CONDA" config --set channel_priority strict >/dev/null 2>&1 || true

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
# 9. Install remaining Python dependencies
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

# scikit-learn >=1.6 removes the "squared" argument used by
# the original CKGConv logger. 1.5.2 keeps the original code working.
log "Pinning scikit-learn to a CKGConv-compatible version"

run_in_env python -m pip install \
    "scikit-learn==1.5.2"

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
# 11. Final compatibility pin
# ------------------------------------------------------------

# IMPORTANT:
# Some packages may pull NumPy 2.x during installation. PyTorch 2.1.2
# was built against NumPy 1.x and its Tensor.numpy() bridge fails with
# NumPy 2.x in this environment. Keep this as the LAST package pin.
log "Pinning NumPy 1.26.4 for PyTorch 2.1.2 compatibility"

run_in_env python -m pip install \
    --force-reinstall \
    --no-deps \
    "numpy==1.26.4"

# ------------------------------------------------------------
# 12. Verification
# ------------------------------------------------------------

log "Verifying installation"

run_in_env python - <<'PY'
import sys

import numpy as np
import sklearn
import torch
import torch_geometric

print("=" * 70)
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("PyTorch Geometric:", torch_geometric.__version__)
print("CUDA available:", torch.cuda.is_available())

if np.__version__ != "1.26.4":
    raise RuntimeError(
        f"Expected NumPy 1.26.4, found {np.__version__}. "
        "PyTorch 2.1.2 requires the NumPy 1.x compatibility baseline used here."
    )

if sklearn.__version__ != "1.5.2":
    raise RuntimeError(
        f"Expected scikit-learn 1.5.2, found {sklearn.__version__}."
    )

# Explicitly verify the failure point we encountered during ZINC logging.
torch_numpy_test = torch.tensor([1.0]).numpy()
print("Torch -> NumPy bridge:", torch_numpy_test)

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

print("=" * 70)
print("CKGConv environment is ready.")
PY

# Check for dependency inconsistencies after all pins.
log "Running pip dependency check"

if ! run_in_env python -m pip check; then
    warn "pip check reported a dependency inconsistency. Review the output above before a long experiment."
fi

# Save a snapshot of the actual working environment for reproducibility.
log "Saving environment snapshots"

run_in_env python -m pip freeze > "$HOME/ckgconv-pip-freeze.txt"
"$CONDA" list -n "$ENV_NAME" > "$HOME/ckgconv-conda-list.txt"

# ------------------------------------------------------------
# 13. Final instructions
# ------------------------------------------------------------

log "Installation completed successfully"

echo
echo "Repository:"
echo "  $REPO_DIR"
echo
echo "Environment snapshots:"
echo "  $HOME/ckgconv-pip-freeze.txt"
echo "  $HOME/ckgconv-conda-list.txt"
echo
echo "To activate the environment:"
echo "  source \"$CONDA_DIR/etc/profile.d/conda.sh\""
echo "  conda activate \"$ENV_NAME\""
echo "  cd \"$REPO_DIR\""
echo
echo "Or run without activation:"
echo "  \"$CONDA\" run --no-capture-output -n \"$ENV_NAME\" python \"$REPO_DIR/main.py\" --help"
echo
