#!/bin/bash
# install_pytorch.sh - Install PyTorch for the appropriate backend

BACKEND=${1:-auto}

if [ "$BACKEND" = "auto" ]; then
    if command -v rocminfo &> /dev/null; then
        BACKEND="rocm"
    elif [[ "$(uname)" == "Darwin" ]]; then
        BACKEND="mps"
    elif command -v nvidia-smi &> /dev/null; then
        BACKEND="cuda"
    else
        BACKEND="cpu"
    fi
fi

case $BACKEND in
    rocm)
        echo "Installing PyTorch with ROCm support..."
        # ROCm 7.2 wheels for Python 3.12 on Ubuntu 24.04
        uv pip install --no-cache-dir \
            "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl" \
            "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl" \
            "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" \
            "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl"
        
        # WSL-specific: Update runtime lib
        if grep -qi microsoft /proc/version 2>/dev/null; then
            echo "Detected WSL - updating HSA runtime..."
            TORCH_LIB=$(uv run python -c "import torch; print(torch.__path__[0])")/lib
            rm -f "${TORCH_LIB}"/libhsa-runtime64.so*
        fi
        ;;
    cuda)
        echo "Installing PyTorch with CUDA support..."
        uv pip install torch torchvision torchaudio
        ;;
    mps|cpu)
        echo "Installing PyTorch (CPU/MPS)..."
        uv pip install torch torchvision torchaudio
        ;;
    *)
        echo "Unknown backend: $BACKEND"
        echo "Usage: $0 [rocm|cuda|mps|cpu|auto]"
        exit 1
        ;;
esac

echo "Done! Verify with: uv run python -c 'import torch; print(torch.cuda.is_available())'"