"""
Mostly copy-paste from https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/ops/attention/attention_utils.py

Pure PyTorch implementation for NPU compatibility.
Replaces CUDA kernels with PyTorch operations.
"""

import torch
import torch.nn as nn
from torch.autograd import Function, Variable


def _compute_key_offset(
    key_batch_cnt: torch.Tensor,
    index_pair_batch: torch.Tensor
) -> torch.Tensor:
    """
    Compute the starting offset for each query in the key features.

    Args:
        key_batch_cnt: [b], number of keys in each batch
        index_pair_batch: [total_query_num], batch index of each query

    Returns:
        key_offset: [total_query_num], starting index in key features for each query
    """
    device = key_batch_cnt.device
    b = key_batch_cnt.shape[0]
    total_query_num = index_pair_batch.shape[0]

    # Compute cumulative sum of key_batch_cnt to get batch start offsets
    # batch_start_idx[i] = sum(key_batch_cnt[0:i])
    batch_start_idx = torch.cat([
        torch.tensor([0], device=device, dtype=key_batch_cnt.dtype),
        torch.cumsum(key_batch_cnt, dim=0)[:-1]
    ])

    # For each query, get its key start offset
    key_offset = batch_start_idx[index_pair_batch]
    return key_offset


class AttentionWeightComputationPytorch(Function):
    """
    Generate the attention weight matrix based on:
        * the generated attention pair index (total_query_num, local_size);
        * query features (total_query_num, nhead, hdim)
        * key features (total_key_num, nhead, hdim)
    Generate the attention weight matrix.
        * (total_query_num, local_size, nhead)

    Pure PyTorch implementation with autograd support.
    """

    @staticmethod
    def forward(ctx,
                query_batch_cnt: torch.Tensor,
                key_batch_cnt: torch.Tensor,
                index_pair_batch: torch.Tensor,
                index_pair: torch.Tensor,
                query_features: torch.Tensor,
                key_features: torch.Tensor):
        """
        :param ctx:
        :param query_batch_cnt: A integer tensor with shape [bs], indicating the query amount for each batch.
        :param key_batch_cnt: A integer tensor with shape [bs], indicating the key amount of each batch.
        :param index_pair_batch: A integer tensor with shape [total_query_num], indicating the batch
            index of each query.
        :param index_pair: A integer tensor with shape [total_query_num, local_size]
            We ignore those index whose value is -1.
        :param query_features: A float tensor with shape [total_query_num, nhead, hdim]
        :param key_features: A float tensor with shape [total_key_num, nhead, hdim]
        :return:
            output: A float tensor with shape [total_query_num, local_size, nhead]
        """
        assert query_batch_cnt.is_contiguous()
        assert key_batch_cnt.is_contiguous()
        assert index_pair_batch.is_contiguous()
        assert index_pair.is_contiguous()
        assert query_features.is_contiguous()
        assert key_features.is_contiguous()

        device = query_features.device
        dtype = query_features.dtype
        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = key_features.size()

        # Need to ensure that every tensor in query features have an output.
        assert total_query_num == query_features.shape[0]

        # Compute key offset for each query (batch-aware)
        key_offset = _compute_key_offset(key_batch_cnt, index_pair_batch)

        # Initialize output tensor
        output = torch.zeros(total_query_num, local_size, nhead,
                             device=device, dtype=dtype)

        # Process each query-key pair
        for query_idx in range(total_query_num):
            offset = key_offset[query_idx]
            query_feat = query_features[query_idx]  # [nhead, hdim]

            for local_idx in range(local_size):
                key_local_idx = index_pair[query_idx, local_idx]

                # Skip invalid indices (-1)
                if key_local_idx == -1:
                    continue

                key_idx = offset + key_local_idx
                key_feat = key_features[key_idx]  # [nhead, hdim]

                # Compute attention weight (dot product) for each head
                attn_weight = (query_feat * key_feat).sum(dim=-1)  # [nhead]
                output[query_idx, local_idx, :] = attn_weight

        # Save context for backward
        ctx.save_for_backward(
            query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair,
            query_features, key_features, key_offset
        )
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: [total_query_num, local_size, nhead]
        Returns:
            grad_query_features:  [total_query_num, nhead, hdim]
            grad_key_features: [total_key_num, nhead, hdim]
        """
        (query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair,
         query_features, key_features, key_offset) = ctx.saved_tensors

        device = query_features.device
        dtype = query_features.dtype
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = key_features.size()

        # Initialize gradient tensors
        grad_query_features = torch.zeros(total_query_num, nhead, hdim,
                                           device=device, dtype=dtype)
        grad_key_features = torch.zeros(total_key_num, nhead, hdim,
                                         device=device, dtype=dtype)

        # Compute gradients using chain rule
        for query_idx in range(total_query_num):
            offset = key_offset[query_idx]
            query_feat = query_features[query_idx]  # [nhead, hdim]

            for local_idx in range(local_size):
                key_local_idx = index_pair[query_idx, local_idx]

                # Skip invalid indices
                if key_local_idx == -1:
                    continue

                key_idx = offset + key_local_idx
                key_feat = key_features[key_idx]  # [nhead, hdim]
                grad = grad_out[query_idx, local_idx]  # [nhead]

                # grad_query += grad * key (broadcasted)
                grad_query_features[query_idx] += grad.unsqueeze(-1) * key_feat
                # grad_key += grad * query (broadcasted)
                grad_key_features[key_idx] += grad.unsqueeze(-1) * query_feat

        return None, None, None, None, grad_query_features, grad_key_features


attention_weight_computation_pytorch = AttentionWeightComputationPytorch.apply


class AttentionValueComputationPytorch(Function):
    """
    Generate the attention result based on:
        * the generated attention pair index (total_query_num, local_size);
        * value features (total_key_num, nhead, hdim)
        * attn_weight (total_query_num, local_size, nhead)
    Generate the attention result.
        * (total_query_num, nhead, hdim)

    Pure PyTorch implementation with autograd support.
    """

    @staticmethod
    def forward(ctx,
                query_batch_cnt: torch.Tensor,
                key_batch_cnt: torch.Tensor,
                index_pair_batch: torch.Tensor,
                index_pair: torch.Tensor,
                attn_weight: torch.Tensor,
                value_features: torch.Tensor):
        """
        :param ctx:
        :param query_batch_cnt: A integer tensor with shape [bs], indicating the query amount for each batch.
        :param key_batch_cnt: A integer tensor with shape [bs], indicating the key amount of each batch.
        :param index_pair_batch: A integer tensor with shape [total_query_num], indicating the batch
            index of each query.
        :param index_pair: A integer tensor with shape [total_query_num, local_size]
            We ignore those index whose value is -1.
        :param attn_weight: A float tensor with shape [total_query_num, local_size, nhead]
        :param value_features: A float tensor with shape [total_key_num, nhead, hdim]
        :return:
            output: A float tensor with shape [total_query_num, nhead, hdim]
        """
        assert query_batch_cnt.is_contiguous()
        assert key_batch_cnt.is_contiguous()
        assert index_pair_batch.is_contiguous()
        assert index_pair.is_contiguous()
        assert attn_weight.is_contiguous()
        assert value_features.is_contiguous()

        device = attn_weight.device
        dtype = attn_weight.dtype
        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = value_features.size()

        # Need to ensure that every tensor in query features have an output.
        assert total_query_num == attn_weight.shape[0]

        # Compute key offset for each query
        key_offset = _compute_key_offset(key_batch_cnt, index_pair_batch)

        # Initialize output tensor
        output = torch.zeros(total_query_num, nhead, hdim,
                             device=device, dtype=dtype)

        # Accumulate weighted value features
        for query_idx in range(total_query_num):
            offset = key_offset[query_idx]

            for local_idx in range(local_size):
                key_local_idx = index_pair[query_idx, local_idx]

                # Skip invalid indices
                if key_local_idx == -1:
                    continue

                key_idx = offset + key_local_idx
                value_feat = value_features[key_idx]  # [nhead, hdim]
                weight = attn_weight[query_idx, local_idx]  # [nhead]

                # output[query_idx] += weight * value_feat (broadcasted)
                output[query_idx] += weight.unsqueeze(-1) * value_feat

        # Save context for backward
        ctx.save_for_backward(
            query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair,
            attn_weight, value_features, key_offset
        )
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: [total_query_num, nhead, hdim]
        Returns:
            grad_attn_weight:  [total_query_num, local_size, nhead]
            grad_value_features: [total_key_num, nhead, hdim]
        """
        (query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair,
         attn_weight, value_features, key_offset) = ctx.saved_tensors

        device = grad_out.device
        dtype = grad_out.dtype
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = value_features.size()

        # Initialize gradient tensors
        grad_attn_weight = torch.zeros(total_query_num, local_size, nhead,
                                        device=device, dtype=dtype)
        grad_value_features = torch.zeros(total_key_num, nhead, hdim,
                                           device=device, dtype=dtype)

        # Compute gradients using chain rule
        for query_idx in range(total_query_num):
            offset = key_offset[query_idx]
            g = grad_out[query_idx]  # [nhead, hdim]

            for local_idx in range(local_size):
                key_local_idx = index_pair[query_idx, local_idx]

                # Skip invalid indices
                if key_local_idx == -1:
                    continue

                key_idx = offset + key_local_idx
                value_feat = value_features[key_idx]  # [nhead, hdim]
                weight = attn_weight[query_idx, local_idx]  # [nhead]

                # grad_attn_weight: element-wise product and sum along hdim
                grad_attn_weight[query_idx, local_idx] = (g * value_feat).sum(dim=-1)
                # grad_value_features: broadcast weight across hdim
                grad_value_features[key_idx] += g * weight.unsqueeze(-1)

        return None, None, None, None, grad_attn_weight, grad_value_features


attention_value_computation_pytorch = AttentionValueComputationPytorch.apply


# Export functions for compatibility
attention_weight_computation = attention_weight_computation_pytorch
attention_value_computation = attention_value_computation_pytorch
