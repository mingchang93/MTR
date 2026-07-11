# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Li Jiang, Shaoshuai Shi
# All Rights Reserved

"""
KNN operations module - auto-selects the best available implementation.

Controlled by the MTR_USE_CUDA environment variable:
  MTR_USE_CUDA=1  -> Use CUDA custom kernel (GPU)
  MTR_USE_CUDA=0  -> Use mx_driving NPU-accelerated KNN (NPU)
  MTR_USE_CUDA unset -> Auto-detect: CUDA if available, otherwise mx_driving

Fallback: Pure PyTorch when neither CUDA nor mx_driving is available.
"""

import os
from mtr.ops.attention import _detect_use_cuda

_USE_CUDA = _detect_use_cuda()
_knn_batch = None
_knn_batch_mlogk = None
_backend_name = None

_MTR_USE_PYTORCH_KNN = os.environ.get('MTR_USE_PYTORCH_KNN', '1')

if _USE_CUDA:
    try:
        # Import knn_cuda first so it's registered in sys.modules before
        # knn_utils tries to import it (avoids circular import)
        import mtr.ops.knn.knn_cuda  # noqa: F401
        from .knn_utils import knn_batch, knn_batch_mlogk
        _knn_batch = knn_batch
        _knn_batch_mlogk = knn_batch_mlogk
        _backend_name = 'CUDA'
    except ImportError as e:
        print(f"[KNN] CUDA KNN not available ({e}), falling back...")

# MTR_USE_CUDA=0: Use mx_driving NPU-accelerated KNN or pure PyTorch fallback
else:
    if _MTR_USE_PYTORCH_KNN == '1':
        from .knn_utils_pytorch import knn_batch, knn_batch_mlogk
        _knn_batch = knn_batch
        _knn_batch_mlogk = knn_batch_mlogk
        _backend_name = 'PyTorch'
    else:
        try:
            from .knn_utils_mx_driving import knn_batch, knn_batch_mlogk
            _knn_batch = knn_batch
            _knn_batch_mlogk = knn_batch_mlogk
            _backend_name = 'mx_driving (NPU)'
        except ImportError as e:
            print(f"[KNN] mx_driving KNN not available ({e}), falling back...")

    # Fallback: if mx_driving failed, use pure PyTorch
    if _knn_batch is None:
        from .knn_utils_pytorch import knn_batch, knn_batch_mlogk
        _knn_batch = knn_batch
        _knn_batch_mlogk = knn_batch_mlogk
        _backend_name = 'PyTorch'

print(f"[KNN] Using {_backend_name} implementation")

__all__ = ['knn_batch', 'knn_batch_mlogk']
