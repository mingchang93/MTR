"""
Mostly copy-paste from https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/ops/attention/attention_utils_v2.py

Pure PyTorch implementation for NPU compatibility.
Replaces CUDA kernels with vectorized PyTorch operations (no Python for-loops).
Uses nn.Module so that autograd handles backward automatically.
"""

import json
import os
import time
import torch
import torch.nn as nn


# --------------- Data dump utilities for precision validation ---------------
_DUMP_DIR = os.environ.get('MTR_DUMP_ATTENTION_DATA', '')
_DUMP_MAX_STEPS = int(os.environ.get('MTR_DUMP_ATTENTION_MAX_STEPS', '10'))
_REQUIRE_CUDA = os.environ.get('MTR_REQUIRE_PYTORCH_ATTENTION_CUDA', '0') == '1'
_DUMP_RUN_DIR = ''
_dump_call_counter = 0
_dump_op_counters = {}

if _DUMP_DIR:
    _DUMP_RUN_DIR = os.path.join(
        _DUMP_DIR, f'run_{time.strftime("%Y%m%d-%H%M%S")}_pid{os.getpid()}')


def _clone_dump_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _dump_shape(value):
    if torch.is_tensor(value):
        return list(value.shape)
    return None


def _dump_attention_call(op_name, backend, phase, counter, data_dict):
    if not _DUMP_RUN_DIR:
        return

    os.makedirs(_DUMP_RUN_DIR, exist_ok=True)
    payload = {k: _clone_dump_value(v) for k, v in data_dict.items()}
    filename = f'{counter:06d}_{op_name}_{backend}_{phase}.pt'
    filepath = os.path.join(_DUMP_RUN_DIR, filename)
    torch.save(payload, filepath)

    record = {
        'filename': filename,
        'op_name': op_name,
        'backend': backend,
        'phase': phase,
        'counter': counter,
        'keys': list(data_dict.keys()),
        'shapes': {k: _dump_shape(v) for k, v in data_dict.items()},
    }
    manifest_path = os.path.join(_DUMP_RUN_DIR, 'manifest.jsonl')
    with open(manifest_path, 'a') as f:
        f.write(json.dumps(record) + '\n')

    print(f'[DUMP] Saved {filepath}')


def _next_dump_counter(op_name):
    global _dump_call_counter
    op_counter = _dump_op_counters.get(op_name, 0)
    if op_counter >= _DUMP_MAX_STEPS:
        return None
    _dump_op_counters[op_name] = op_counter + 1
    counter = _dump_call_counter
    _dump_call_counter += 1
    return counter
# ---------------------------------------------------------------------------


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

    CUDA-to-PyTorch line mapping:
        - attention_weight_computation_kernel_v2.cu:55-60 and :145-150
          attention_value_computation_kernel_v2.cu:40-44 and :134-138
          compute batch_idx and key_start_idx with a for-loop over key_batch_cnt.
        - The PyTorch statements below replace that loop with prefix sums and
          vectorized indexing: batch_start_idx[index_pair_batch].

    Intermediate variable correspondence:
        CUDA batch_idx              <-> PyTorch index_pair_batch
        CUDA key_start_idx per query <-> PyTorch key_offset
        CUDA key_batch_cnt[i]       <-> PyTorch key_batch_cnt
    """
    device = key_batch_cnt.device
    # CUDA key_start_idx loop -> PyTorch prefix starts for all batches.
    batch_start_idx = torch.cat([
        torch.tensor([0], device=device, dtype=key_batch_cnt.dtype),
        torch.cumsum(key_batch_cnt, dim=0)[:-1]
    ])
    # CUDA key_start_idx selected by batch_idx -> vectorized per-query offset.
    key_offset = batch_start_idx[index_pair_batch]
    return key_offset


class AttentionWeightComputationModule(nn.Module):
    """
    Compute attention weights: dot product between query and key features.

    Input:
        query_features: [total_query_num, nhead, hdim]
        key_features:   [total_key_num, nhead, hdim]
        index_pair:     [total_query_num, local_size]  (-1 = invalid)
    Output:
        [total_query_num, local_size, nhead]

    Uses standard PyTorch ops; backward is handled by autograd.

    CUDA-to-PyTorch line mapping for forward:
        1. cu:32-39 block/thread ids and bounds -> py: tensor shapes.
        2. cu:41-47 shared_query_features load -> py: query_features.unsqueeze(1).
        3. cu:49-52 invalid index early return -> py: valid_mask and final masking.
        4. cu:55-60 key_start_idx -> py: _compute_key_offset + abs_key_idx.
        5. cu:62-64 key pointer arithmetic -> py: torch.gather.
        6. cu:66-70 hdim dot-product loop -> py: broadcast multiply + sum(dim=-1).

    Intermediate variable correspondence:
        CUDA shared_query_features[i] <-> PyTorch query_features.unsqueeze(1)
        CUDA index_pair[index]        <-> PyTorch index_pair / valid_mask
        CUDA key_start_idx            <-> PyTorch key_offset / abs_key_idx
        CUDA key_features pointer     <-> PyTorch gathered_keys
        CUDA attn_weight/output[0]    <-> PyTorch output
    """

    def forward(self,
                query_batch_cnt: torch.Tensor,
                key_batch_cnt: torch.Tensor,
                index_pair_batch: torch.Tensor,
                index_pair: torch.Tensor,
                query_features: torch.Tensor,
                key_features: torch.Tensor) -> torch.Tensor:
        if _REQUIRE_CUDA:
            assert query_features.is_cuda and key_features.is_cuda

        b = query_batch_cnt.shape[0]
        # cu:32-39 -> PyTorch names the CUDA block/thread axes as tensor dimensions.
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = key_features.size()

        dump_counter = _next_dump_counter('weight_forward') if _DUMP_DIR else None
        output = query_features.new_zeros(total_query_num, local_size, nhead)
        dump_data = {
            'backend': 'pytorch',
            'op_name': 'weight_forward',
            'phase': 'before',
            'b': b,
            'total_query_num': total_query_num,
            'local_size': local_size,
            'total_key_num': total_key_num,
            'nhead': nhead,
            'hdim': hdim,
            'query_batch_cnt': query_batch_cnt,
            'key_batch_cnt': key_batch_cnt,
            'index_pair_batch': index_pair_batch,
            'index_pair': index_pair,
            'query_features': query_features,
            'key_features': key_features,
            'output': output,
        }
        if dump_counter is not None:
            _dump_attention_call('weight_forward', 'pytorch', 'before', dump_counter, dump_data)

        # cu:55-60 -> vectorized key_start_idx for every query.
        key_offset = _compute_key_offset(key_batch_cnt, index_pair_batch)

        # cu:60 -> local key index plus batch offset gives absolute key index.
        abs_key_idx = index_pair + key_offset.unsqueeze(-1)

        # cu:49-52 -> CUDA skips invalid pairs; PyTorch masks them after dense ops.
        valid_mask = (index_pair != -1).to(query_features.dtype)  # [total_query_num, local_size]
        safe_idx = abs_key_idx.clamp(min=0)

        # cu:62-64 -> pointer arithmetic becomes gather over key_features dim 0.
        gather_idx = safe_idx.reshape(-1, 1, 1).expand(-1, nhead, hdim)
        gathered_keys = torch.gather(key_features, 0, gather_idx).reshape(
            total_query_num, local_size, nhead, hdim)

        # cu:41-47 and cu:66-70 -> shared query cache + hdim loop become broadcast dot product.
        output = (query_features.unsqueeze(1) * gathered_keys).sum(dim=-1)

        # cu:49-52 -> invalid CUDA threads leave pre-zeroed output unchanged.
        output = output * valid_mask.unsqueeze(-1)

        if dump_counter is not None:
            dump_data['phase'] = 'after'
            dump_data['output'] = output
            _dump_attention_call('weight_forward', 'pytorch', 'after', dump_counter, dump_data)

        return output


class AttentionValueComputationModule(nn.Module):
    """
    Compute attention output: weighted sum of value features.

    Input:
        attn_weight:     [total_query_num, local_size, nhead]
        value_features:  [total_key_num, nhead, hdim]
        index_pair:      [total_query_num, local_size]  (-1 = invalid)
    Output:
        [total_query_num, nhead, hdim]

    Uses standard PyTorch ops; backward is handled by autograd.

    CUDA-to-PyTorch line mapping for forward:
        1. cu:32-37 block/thread ids and bounds -> py: tensor shapes.
        2. cu:40-44 key_start_idx -> py: _compute_key_offset + abs_key_idx.
        3. cu:48-61 shared_attn_weight/shared_value_indices fill -> py:
           valid_mask, safe_idx, torch.gather, and attn_weight tensor itself.
        4. cu:64 output pointer arithmetic -> py: native output layout.
        5. cu:66-72 local_size weighted-sum loop -> py: masked broadcast
           multiply + sum(dim=1).

    Intermediate variable correspondence:
        CUDA shared_attn_weight[i]   <-> PyTorch attn_weight / masked_attn
        CUDA cur_key_idx             <-> PyTorch abs_key_idx
        CUDA shared_value_indices[i] <-> PyTorch safe_idx / valid_mask
        CUDA value_features[...]     <-> PyTorch gathered_values
        CUDA attn_result/output[0]   <-> PyTorch output
    """

    def forward(self,
                query_batch_cnt: torch.Tensor,
                key_batch_cnt: torch.Tensor,
                index_pair_batch: torch.Tensor,
                index_pair: torch.Tensor,
                attn_weight: torch.Tensor,
                value_features: torch.Tensor) -> torch.Tensor:
        if _REQUIRE_CUDA:
            assert attn_weight.is_cuda and value_features.is_cuda

        b = query_batch_cnt.shape[0]
        # cu:32-37 -> PyTorch names the CUDA block/thread axes as tensor dimensions.
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = value_features.size()

        dump_counter = _next_dump_counter('value_forward') if _DUMP_DIR else None
        output = value_features.new_zeros(total_query_num, nhead, hdim)
        dump_data = {
            'backend': 'pytorch',
            'op_name': 'value_forward',
            'phase': 'before',
            'b': b,
            'total_query_num': total_query_num,
            'local_size': local_size,
            'total_key_num': total_key_num,
            'nhead': nhead,
            'hdim': hdim,
            'query_batch_cnt': query_batch_cnt,
            'key_batch_cnt': key_batch_cnt,
            'index_pair_batch': index_pair_batch,
            'index_pair': index_pair,
            'attn_weight': attn_weight,
            'value_features': value_features,
            'output': output,
        }
        if dump_counter is not None:
            _dump_attention_call('value_forward', 'pytorch', 'before', dump_counter, dump_data)

        # cu:40-44 -> vectorized key_start_idx and absolute value indices.
        key_offset = _compute_key_offset(key_batch_cnt, index_pair_batch)

        # cu:59-60 -> local value index plus batch offset gives absolute value index.
        abs_key_idx = index_pair + key_offset.unsqueeze(-1)
        valid_mask = (index_pair != -1).to(attn_weight.dtype)  # [total_query_num, local_size]
        safe_idx = abs_key_idx.clamp(min=0)

        # cu:54-60 and cu:69-70 -> shared_value_indices plus pointer arithmetic become gather.
        gather_idx = safe_idx.reshape(-1, 1, 1).expand(-1, nhead, hdim)
        gathered_values = torch.gather(value_features, 0, gather_idx).reshape(
            total_query_num, local_size, nhead, hdim)

        # cu:51-52 and cu:68 -> shared_attn_weight and invalid skip become masked_attn.
        masked_attn = attn_weight * valid_mask.unsqueeze(-1)  # [total_query_num, local_size, nhead]

        # cu:66-72 -> local_size loop becomes broadcast multiply and sum over local_size.
        output = (masked_attn.unsqueeze(-1) * gathered_values).sum(dim=1)

        if dump_counter is not None:
            dump_data['phase'] = 'after'
            dump_data['output'] = output
            _dump_attention_call('value_forward', 'pytorch', 'after', dump_counter, dump_data)

        return output


# Module instances (callable, same signature as original Function.apply)
attention_weight_computation_pytorch = AttentionWeightComputationModule()
attention_value_computation_pytorch = AttentionValueComputationModule()

# Export for compatibility
attention_weight_computation = attention_weight_computation_pytorch
attention_value_computation = attention_value_computation_pytorch
