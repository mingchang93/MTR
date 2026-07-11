# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Li Jiang, Shaoshuai Shi
# All Rights Reserved

"""
KNN operations using mx_driving (DrivingSDK) NPU-accelerated KNN operator.

This module adapts mx_driving's batched KNN interface to match MTR's flattened
KNN interface (knn_batch_mlogk), enabling seamless NPU acceleration.

Key differences bridged:
  - MTR uses flattened (n, 3) + batch_offsets; mx_driving uses [B, N, 3] batched
  - Variable-length batches are handled by padding to max_n per batch
  - Invalid neighbors (dist2 >= 1e10 from mx_driving) are mapped to -1
"""

import torch
import torch_npu
import mx_driving
from torch.autograd import Function


class KNNBatchMxDriving(Function):
    """
    K-Nearest Neighbors using mx_driving NPU-accelerated operator.

    Adapts MTR's flattened interface to mx_driving's batched interface.
    For each point in xyz, find the k nearest neighbors in query_xyz
    within the same batch.

    Args:
        xyz: (n, 3) float - input points (n total points across all batches)
        query_xyz: (m, 3) float - query points (m total query points across all batches)
        batch_idxs: (n) int - batch index for each point in xyz
        query_batch_offsets: (B+1) int - cumulative offsets for query points in each batch
        k: int - number of nearest neighbors to find

    Returns:
        idx: (n, k) int - indices of k nearest neighbors in query_xyz (relative to each batch)
    """

    @staticmethod
    def forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        assert xyz.is_contiguous()
        assert query_xyz.is_contiguous()
        assert batch_idxs.is_contiguous()
        assert query_batch_offsets.is_contiguous()
        assert k > 0 and k < 100, f"k must be in (0, 100), got {k}"

        device = xyz.device
        B = query_batch_offsets.shape[0] - 1  # number of batches
        n = xyz.shape[0]  # total number of input points

        # Compute per-batch point counts and max count for padding
        batch_cnt = query_batch_offsets[1:] - query_batch_offsets[:-1]  # [B]
        max_n = batch_cnt.max().item()

        # Build padded [B, max_n, 3] tensor for query points
        # Padding coordinates are set to 1e8 so padded points are far from
        # real points and won't be selected as neighbors
        padded_query_xyz = torch.full(
            (B, max_n, 3), 1e8, dtype=query_xyz.dtype, device=device
        )
        for b in range(B):
            start = query_batch_offsets[b].item()
            end = query_batch_offsets[b + 1].item()
            n_b = end - start
            if n_b > 0:
                padded_query_xyz[b, :n_b] = query_xyz[start:end]

        # For xyz (key points), we also need to build padded version.
        # Since xyz and query_xyz may differ, compute xyz offsets from batch_idxs.
        # But in MTR's usage, xyz == query_xyz (self-query), so we reuse the same
        # padded tensor for both.
        # Handle the general case where xyz != query_xyz:
        if xyz.data_ptr() == query_xyz.data_ptr() and xyz.shape == query_xyz.shape:
            # Self-query: same tensor, reuse padded version
            padded_xyz = padded_query_xyz
        else:
            # Different key/query: need separate padding for xyz
            # Compute per-batch key counts from batch_idxs
            key_batch_cnt = torch.zeros(B, dtype=torch.int64, device=device)
            key_batch_cnt.scatter_add_(0, batch_idxs.long(), torch.ones_like(batch_idxs, dtype=torch.int64))
            max_key = key_batch_cnt.max().item()

            # Build key_start_offsets (like query_batch_offsets for keys)
            key_offsets = torch.zeros(B + 1, dtype=torch.int32, device=device)
            key_offsets[1:] = torch.cumsum(key_batch_cnt, dim=0)

            padded_xyz = torch.full(
                (B, max_key, 3), 1e8, dtype=xyz.dtype, device=device
            )
            # Scatter key points into padded tensor
            # We need per-batch local indices
            local_idx = torch.arange(n, device=device) - key_offsets[batch_idxs].long()
            b_idx = batch_idxs.long()
            padded_xyz[b_idx, local_idx] = xyz

        # Call mx_driving._C.knn to get both dist2 and idx
        # C API expects xyz in [B, 3, N] format, center_xyz in [B, npoint, 3] format
        xyz_t = padded_xyz.transpose(2, 1).contiguous()  # [B, 3, max_n] or [B, 3, max_key]
        dist2, idx = mx_driving._C.knn(xyz_t, padded_query_xyz, k, True)
        # dist2: [B, max_n, k], idx: [B, max_n, k]

        # Replace invalid neighbor indices with -1
        # mx_driving uses dist2 >= 1e10 as sentinel for invalid/missing neighbors
        idx[dist2 >= 1e10] = -1

        # Extract results for valid (non-padded) query points only
        # For each batch b, only the first n_b rows are valid query points
        result = torch.zeros(n, k, dtype=torch.int32, device=device)
        for b in range(B):
            start = query_batch_offsets[b].item()
            end = query_batch_offsets[b + 1].item()
            n_b = end - start
            if n_b > 0:
                result[start:end] = idx[b, :n_b, :]

        return result

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


class KNNBatchMlogKMxDriving(Function):
    """
    K-Nearest Neighbors using max-heap via mx_driving NPU-accelerated operator.

    Same interface as KNNBatchMxDriving. The mx_driving backend already uses
    an optimized algorithm internally, so both classes delegate to the same
    NPU operator.

    Args:
        xyz: (n, 3) float - input points
        query_xyz: (m, 3) float - query points
        batch_idxs: (n) int - batch index for each point in xyz
        query_batch_offsets: (B+1) int - cumulative offsets for query points
        k: int - number of nearest neighbors (max 99, mx_driving limit)

    Returns:
        idx: (n, k) int - indices of k nearest neighbors (relative to each batch)
    """

    @staticmethod
    def forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        # Delegate to KNNBatchMxDriving - mx_driving uses the same optimized
        # NPU operator regardless of algorithm variant
        return KNNBatchMxDriving.forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


knn_batch = KNNBatchMxDriving.apply
knn_batch_mlogk = KNNBatchMlogKMxDriving.apply
