# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Li Jiang, Shaoshuai Shi
# All Rights Reserved

"""
Pure PyTorch implementation of KNN operations.

Uses vectorized distance computation (||x||^2 + ||y||^2 - 2 * x * y^T) with torch.mm
and torch.topk to replace the original Python-loop-based approach. Chunked
processing limits peak memory for large batches.

Performance: O(N*M) distance computation runs on-device via BLAS (torch.mm);
top-k is also fully vectorized. Expected wall time drops from hours to seconds
compared to the original per-element Python loop implementation.

Environment variables:
  MTR_KNN_CHUNK_SIZE: points per chunk (default 4096). Lower -> less memory.
"""

import os
import torch
from torch.autograd import Function

_CHUNK_SIZE = int(os.environ.get('MTR_KNN_CHUNK_SIZE', '4096'))


def _squared_distance_matrix(x, y):
    """
    Compute squared Euclidean distance matrix: ||x||^2 + ||y||^2 - 2 * x * y^T.

    Uses torch.mm (BLAS-accelerated) for the cross term, avoiding per-element
    Python loops. Equivalent to torch.cdist(x, y)^2 but avoids the sqrt.

    Args:
        x: (N, 3) float tensor
        y: (M, 3) float tensor

    Returns:
        dist_sq: (N, M) squared Euclidean distances
    """
    x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)   # (N, 1)
    y_norm_sq = (y ** 2).sum(dim=1, keepdim=True)   # (M, 1)
    cross = torch.mm(x, y.T)                         # (N, M), BLAS-accelerated
    dist_sq = x_norm_sq + y_norm_sq.T - 2.0 * cross
    return dist_sq.clamp(min=0.0)


def _knn_vectorized(xyz_b, query_xyz_b, k, is_self_query=False):
    """
    Vectorized KNN for points within a single batch.

    Computes the full pairwise distance matrix (chunked if necessary), then
    uses torch.topk for efficient k-nearest-neighbor selection. For self-query
    (xyz_b is query_xyz_b semantically), the diagonal is set to inf to exclude self.

    IMPORTANT: ``is_self_query`` must be determined by the CALLER using the
    ORIGINAL tensors, because ``xyz[mask]`` (boolean indexing) creates a copy
    with new storage -- data_ptr() comparison inside this function would fail.
    The MTR encoder always calls with xyz == query_xyz (same tensor).

    Args:
        xyz_b: (n_key, 3) key points (may be a copy from boolean indexing)
        query_xyz_b: (n_query, 3) query points (may be a slice/view)
        k: number of nearest neighbors
        is_self_query: True when the ORIGINAL xyz and query_xyz are the same tensor

    Returns:
        idx: (n_key, k) int32 indices of k nearest neighbors (relative to query_xyz_b)
    """
    n_key = xyz_b.shape[0]
    n_query = query_xyz_b.shape[0]
    device = xyz_b.device

    if n_query == 0 or n_key == 0:
        return torch.full((n_key, k), -1, dtype=torch.int32, device=device)

    actual_k = min(k, n_query)

    # ---- small enough -> compute directly ----
    if n_key <= _CHUNK_SIZE:
        dist_sq = _squared_distance_matrix(xyz_b, query_xyz_b)    # (n_key, n_query)

        if is_self_query and n_key == n_query:
            dist_sq.fill_diagonal_(float('inf'))

        _, topk_idx = torch.topk(dist_sq, actual_k, dim=1, largest=False)
        return _pad_or_return(topk_idx, k, device)

    # ---- large batch -> chunked computation ----
    all_idx = []
    for i in range(0, n_key, _CHUNK_SIZE):
        chunk = xyz_b[i:i + _CHUNK_SIZE]                         # (C, 3)
        dist_chunk = _squared_distance_matrix(chunk, query_xyz_b)  # (C, n_query)

        if is_self_query and n_key == n_query:
            # Build diagonal mask for this chunk: row j <-> column (i + j)
            row_indices = torch.arange(i, i + chunk.shape[0], device=device)
            col_indices = torch.arange(n_query, device=device)
            diag_mask = (col_indices.unsqueeze(0) == row_indices.unsqueeze(1))
            dist_chunk[diag_mask] = float('inf')

        _, topk_chunk = torch.topk(dist_chunk, actual_k, dim=1, largest=False)
        all_idx.append(topk_chunk)

    topk_idx = torch.cat(all_idx, dim=0)                         # (n_key, actual_k)
    return _pad_or_return(topk_idx, k, device)


def _pad_or_return(topk_idx, k, device):
    """Pad result with -1 if actual_k < k."""
    if topk_idx.shape[1] < k:
        n_key = topk_idx.shape[0]
        padded = torch.full((n_key, k), -1, dtype=torch.int32, device=device)
        padded[:, :topk_idx.shape[1]] = topk_idx
        return padded
    return topk_idx.int()


class KNNBatchPytorch(Function):
    """
    K-Nearest Neighbors for batched point clouds (vectorized).

    For each batch independently, computes pairwise distances via the squared
    expansion formula and selects top-k via torch.topk. Processes batches
    sequentially but each batch's distance matrix is fully vectorized.

    Args:
        xyz: (n, 3) float - input points (n total points across all batches)
        query_xyz: (m, 3) float - query points (m total query points across all batches)
        batch_idxs: (n) int - batch index for each point in xyz
        query_batch_offsets: (B+1) int - cumulative offsets for query points in each batch
        k: int - number of nearest neighbors to find

    Returns:
        idx: (n, k) int32 - indices of k nearest neighbors (relative to each batch's query start)
    """

    @staticmethod
    def forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        assert xyz.is_contiguous()
        assert query_xyz.is_contiguous()
        assert batch_idxs.is_contiguous()
        assert query_batch_offsets.is_contiguous()

        device = xyz.device
        B = query_batch_offsets.shape[0] - 1
        n = xyz.shape[0]

        # ---- detect self-query BEFORE boolean indexing (which creates copies) ----
        is_self_query = (xyz.data_ptr() == query_xyz.data_ptr() and
                         xyz.shape == query_xyz.shape)

        # ---- single CPU sync: fetch all batch boundaries ----
        q_offsets = query_batch_offsets.cpu().tolist()

        result = torch.full((n, k), -1, dtype=torch.int32, device=device)

        for b in range(B):
            q_start = q_offsets[b]
            q_end = q_offsets[b + 1]
            if q_start == q_end:
                continue   # empty batch

            # select xyz points belonging to batch b (device-side mask)
            # NOTE: boolean indexing creates a COPY -- is_self_query was detected above
            mask = (batch_idxs == b)
            if not mask.any().item():
                continue

            xyz_b = xyz[mask]                            # (n_b, 3) -- COPY
            query_b = query_xyz[q_start:q_end]           # (m_b, 3) -- slice/view

            idx_b = _knn_vectorized(xyz_b, query_b, k, is_self_query=is_self_query)
            result[mask] = idx_b

        return result

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


class KNNBatchMlogKPytorch(Function):
    """
    K-Nearest Neighbors using vectorized torch.topk (replaces manual max-heap).

    Uses the same vectorized engine as KNNBatchPytorch. The 'MlogK' name is
    retained for API compatibility; the original CUDA version used a max-heap
    for O(m log k) complexity, but the vectorized approach is O(m) per point
    with BLAS-accelerated distance computation -- asymptotically superior.

    Args:
        xyz: (n, 3) float - input points
        query_xyz: (m, 3) float - query points
        batch_idxs: (n) int - batch index for each point in xyz
        query_batch_offsets: (B+1) int - cumulative offsets for query points
        k: int - number of nearest neighbors (max 128, API contract from CUDA version)

    Returns:
        idx: (n, k) int32 - indices of k nearest neighbors (relative to each batch)
    """

    @staticmethod
    def forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        assert k <= 128, f"k must be <= 128 (CUDA API contract), got {k}"
        # Delegate to the same vectorized path
        return KNNBatchPytorch.forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


knn_batch = KNNBatchPytorch.apply
knn_batch_mlogk = KNNBatchMlogKPytorch.apply
