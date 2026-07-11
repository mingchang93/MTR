# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Li Jiang, Shaoshuai Shi
# All Rights Reserved

"""
Attention operations module.

Supports both CUDA and pure PyTorch implementations.
Selection priority: MTR_USE_CUDA env var > auto-detection > PyTorch fallback.
"""

import os


def _detect_use_cuda():
    """
    Determine whether to use CUDA attention implementation.

    Priority:
      1. MTR_USE_CUDA env var if explicitly set ('1' or '0')
      2. Auto-detect from torch.cuda.is_available()
      3. Fall back to PyTorch (non-CUDA) implementation

    Returns:
        bool: True if CUDA implementation should be used
    """
    env_val = os.environ.get('MTR_USE_CUDA')
    if env_val is not None:
        return env_val == '1'

    # Auto-detect: check if CUDA is available at runtime
    try:
        import torch
        if torch.cuda.is_available():
            print("Auto-detected CUDA device, using CUDA attention implementation")
            return True
    except ImportError:
        pass

    # Default: use PyTorch (NPU-compatible)
    return False


USE_CUDA = _detect_use_cuda()
PYTORCH_ATTENTION_MODE = os.environ.get('MTR_PYTORCH_ATTENTION_MODE', 'autograd')

# Always import PyTorch implementations (pure Python, always available)
from . import attention_utils_pytorch
from . import attention_utils_v2_pytorch

if USE_CUDA:
    try:
        from . import attention_utils
        from . import attention_utils_v2
        print("Using CUDA attention implementation")
    except ImportError:
        print("CUDA attention not available, falling back to PyTorch")
        attention_utils = attention_utils_pytorch
        attention_utils_v2 = attention_utils_v2_pytorch
else:
    attention_utils = attention_utils_pytorch
    if PYTORCH_ATTENTION_MODE == 'manual':
        from . import attention_utils_v2
        print("Using PyTorch attention implementation with manual backward")
    else:
        attention_utils_v2 = attention_utils_v2_pytorch
        print("Using PyTorch attention implementation (NPU compatible)")

__all__ = {
    'v1': attention_utils,
    'v2': attention_utils_v2,
    'v1_pytorch': attention_utils_pytorch,
    'v2_pytorch': attention_utils_v2_pytorch,
}
