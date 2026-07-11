"""
Mostly copy-paste from https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/ops/attention/attention_utils_v2.py

Supports both CUDA custom ops and PyTorch native fallback.
Set environment variable MTR_USE_CUDA=1 to use CUDA custom ops (default: PyTorch native).
"""

import json
import os
import time
import torch
import torch.nn as nn
from torch.autograd import Function, Variable

_MTR_USE_CUDA = os.environ.get('MTR_USE_CUDA', '0') == '1'

if _MTR_USE_CUDA:
    try:
        from . import attention_cuda
    except ImportError:
        import warnings
        warnings.warn(
            "MTR_USE_CUDA=1 but attention_cuda extension is not available. "
            "Falling back to PyTorch native implementation."
        )
        _MTR_USE_CUDA = False


# --------------- Data dump utilities for precision validation ---------------
_DUMP_DIR = os.environ.get('MTR_DUMP_ATTENTION_DATA', '')
_DUMP_MAX_STEPS = int(os.environ.get('MTR_DUMP_ATTENTION_MAX_STEPS', '10'))
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
    """Save one attention call snapshot for CUDA/PyTorch precision comparison.

    The same data_dict keys must be passed for before/after and for CUDA/PyTorch.
    Files are written under a unique run directory so repeated executions do not
    overwrite older dumps.
    """
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


def _compute_key_offset(key_batch_cnt, index_pair_batch):
    """Compute the starting offset for each query in the key features.

    For each query, compute key_start_idx = sum(key_batch_cnt[0:batch_idx])
    where batch_idx is the batch index of the query.

    This matches the CUDA kernel logic:
        int batch_idx = index_pair_batch[query_idx];
        int key_start_idx = 0;
        for (int i = 0; i < batch_idx; i++){
            key_start_idx += key_batch_cnt[i];
        }

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


# ============== PyTorch native implementations (for non-CUDA path) ==============
# These match the CUDA kernel logic for switch(hdim/local_size) case 64.
# The PyTorch implementations are general and work for any hdim/local_size value,
# but the numerical behavior has been verified against the case 64 CUDA kernels.


def _attention_weight_computation_forward_pytorch(
        query_batch_cnt, key_batch_cnt, index_pair_batch,
        index_pair, query_features, key_features):
    """
    PyTorch native forward for attention weight computation.

    Matches CUDA kernel attention_weight_computation_forward_v2 (case hdim=64):
        output[query_idx, local_key_idx, head_idx] =
            sum_i(query_features[query_idx, head_idx, i] * key_features[abs_key_idx, head_idx, i])
    for valid (index_pair != -1) entries; 0 otherwise.

    In the CUDA kernel:
        - Shared memory sized to hdim=64 is used to cache query features.
        - Dot product is computed over hdim dimension.
        - Invalid positions (index_pair == -1) are skipped, output stays 0.

    Shapes:
        query_features: [total_query_num, nhead, hdim]
        key_features:   [total_key_num, nhead, hdim]
        index_pair:     [total_query_num, local_size]
        output:         [total_query_num, local_size, nhead]

    CUDA-to-PyTorch line mapping, in execution order:
        1. cu:32-39 block/thread ids and bounds -> py: index_pair.size(),
           query_features.shape define the dense tensor axes.
        2. cu:41-47 shared_query_features load -> py: query_features.unsqueeze(1)
           broadcasts query features across local_size.
        3. cu:49-52 invalid index early return -> py: valid_mask and final masking.
        4. cu:55-60 key_start_idx -> py: _compute_key_offset + abs_key_idx.
        5. cu:62-64 key pointer/output pointer arithmetic -> py: torch.gather
           into gathered_keys and native output layout.
        6. cu:66-70 hdim dot-product loop -> py: broadcast multiply + sum(dim=-1).

    Intermediate variable correspondence:
        CUDA query_idx/head_idx/local_key_idx <-> PyTorch tensor axes [Q, H, L]
        CUDA shared_query_features[i]         <-> PyTorch query_features.unsqueeze(1)
        CUDA index_pair[index]               <-> PyTorch index_pair
        CUDA key_start_idx                   <-> PyTorch key_offset / abs_key_idx
        CUDA key_features pointer            <-> PyTorch gathered_keys
        CUDA attn_weight                     <-> PyTorch output before valid_mask
        CUDA output[0]                       <-> PyTorch output
    """
    # cu:32-39 -> PyTorch names the CUDA block/thread axes as tensor dimensions.
    total_query_num, local_size = index_pair.size()
    nhead, hdim = query_features.shape[1], query_features.shape[2]

    # cu:55-60 -> vectorized key_start_idx for every query.
    key_offset = _compute_key_offset(key_batch_cnt, index_pair_batch)

    # cu:60 -> local key index plus batch offset gives absolute key index.
    abs_key_idx = index_pair + key_offset.unsqueeze(-1)

    # cu:49-52 -> CUDA skips invalid pairs; PyTorch masks them after dense ops.
    valid_mask = (index_pair != -1).to(query_features.dtype)  # [total_query_num, local_size]
    safe_idx = abs_key_idx.clamp(min=0)  # avoid negative indices for gather

    # cu:62-64 -> pointer arithmetic becomes gather over key_features dim 0.
    gather_idx = safe_idx.reshape(-1, 1, 1).expand(-1, nhead, hdim)
    gathered_keys = torch.gather(key_features, 0, gather_idx).reshape(
        total_query_num, local_size, nhead, hdim)

    # cu:41-47 and cu:66-70 -> shared query cache + hdim loop become broadcast dot product.
    output = (query_features.unsqueeze(1) * gathered_keys).sum(dim=-1)

    # cu:49-52 -> invalid CUDA threads leave pre-zeroed output unchanged.
    output = output * valid_mask.unsqueeze(-1)

    return output


def _attention_weight_computation_backward_pytorch(
        grad_out, query_batch_cnt, key_batch_cnt, index_pair_batch,
        index_pair, query_features, key_features):
    """
    PyTorch native backward for attention weight computation.

    Matches CUDA kernel attention_weight_computation_backward_v2 (case hdim=64):
        grad_query_features[query_idx, head_idx, i] =
            sum_l(grad_out[query_idx, l, head_idx] * key_features[abs_key_idx_l, head_idx, i])
            (accumulated via shared memory atomicAdd across local_key_idx threads)
        grad_key_features[abs_key_idx, head_idx, i] +=
            grad_out[query_idx, l, head_idx] * query_features[query_idx, head_idx, i]
            (accumulated via global atomicAdd across queries referencing same key)

    In the CUDA kernel with hdim=64:
        - shared_query_features[64] caches query features per (query, head) block.
        - shared_grad_query_features[64] accumulates grad for query via atomicAdd.
        - grad_key_features uses global atomicAdd for concurrent access from different blocks.

    The scatter_add for grad_key_features matches CUDA atomicAdd behavior
    (multiple queries may reference the same key).

    Shapes:
        grad_out:            [total_query_num, local_size, nhead]
        query_features:      [total_query_num, nhead, hdim]
        key_features:        [total_key_num, nhead, hdim]
        grad_query_features: [total_query_num, nhead, hdim]
        grad_key_features:   [total_key_num, nhead, hdim]

    CUDA-to-PyTorch line mapping, in execution order:
        1. cu:125-132 block/thread ids and bounds -> py: tensor shapes.
        2. cu:134-142 shared query/grad buffers -> py: query_features broadcast
           and reduced grad_query_features tensor.
        3. cu:144-150 valid pair and key_start_idx -> py: valid_mask,
           _compute_key_offset, abs_key_idx, safe_idx.
        4. cu:152-155 key/grad pointers and gradient load -> py: gathered_keys
           and masked_grad_out.
        5. cu:156-163 atomicAdd to shared_grad_query_features -> py:
           (masked_grad_out * gathered_keys).sum(dim=1).
        6. cu:160-162 global atomicAdd to grad_key_features -> py:
           grad_key_contrib + scatter_add_.
        7. cu:167-169 copy shared grad to global query grad -> py: the reduced
           grad_query_features result is already in global tensor layout.

    Intermediate variable correspondence:
        CUDA gradient                         <-> PyTorch masked_grad_out
        CUDA shared_query_features[i]         <-> PyTorch query_features.unsqueeze(1)
        CUDA shared_grad_query_features[i]    <-> PyTorch grad_query_features
        CUDA key_features pointer             <-> PyTorch gathered_keys
        CUDA grad_key_features atomic target  <-> PyTorch grad_key_features.scatter_add_
        CUDA index_pair[index]                <-> PyTorch index_pair / valid_mask
        CUDA key_start_idx                    <-> PyTorch key_offset / abs_key_idx
    """
    # cu:125-132 -> PyTorch names the CUDA block/thread axes as tensor dimensions.
    total_query_num, local_size = index_pair.size()
    nhead, hdim = query_features.shape[1], query_features.shape[2]
    total_key_num = key_features.shape[0]

    # cu:144-150 -> vectorized key_start_idx and absolute key indices.
    key_offset = _compute_key_offset(key_batch_cnt, index_pair_batch)
    abs_key_idx = index_pair + key_offset.unsqueeze(-1)  # [total_query_num, local_size]
    valid_mask = (index_pair != -1).to(grad_out.dtype)    # [total_query_num, local_size]
    safe_idx = abs_key_idx.clamp(min=0)

    # cu:144 and cu:155 -> only valid CUDA threads load gradient.
    masked_grad_out = grad_out * valid_mask.unsqueeze(-1)  # [total_query_num, local_size, nhead]

    # cu:152 -> key pointer arithmetic becomes gather over key_features dim 0.
    gather_idx = safe_idx.reshape(-1, 1, 1).expand(-1, nhead, hdim)
    gathered_keys = torch.gather(key_features, 0, gather_idx).reshape(
        total_query_num, local_size, nhead, hdim)

    # cu:156-159 -> shared-memory atomicAdd over local_key_idx becomes sum(dim=1).
    grad_query_features = (masked_grad_out.unsqueeze(-1) * gathered_keys).sum(dim=1)

    # cu:160-162 -> each valid pair contributes to a global key-gradient atomicAdd.
    grad_key_contrib = masked_grad_out.unsqueeze(-1) * query_features.unsqueeze(1)

    grad_key_contrib_flat = grad_key_contrib.reshape(-1, nhead, hdim)
    safe_idx_flat = safe_idx.reshape(-1, 1, 1).expand(-1, nhead, hdim)

    # cu:160-162 -> global atomicAdd is represented by scatter_add_ on key index.
    grad_key_features = torch.zeros(
        total_key_num, nhead, hdim,
        dtype=key_features.dtype, device=key_features.device)
    grad_key_features.scatter_add_(0, safe_idx_flat, grad_key_contrib_flat)

    return grad_query_features, grad_key_features


def _attention_value_computation_forward_pytorch(
        query_batch_cnt, key_batch_cnt, index_pair_batch,
        index_pair, attn_weight, value_features):
    """
    PyTorch native forward for attention value computation.

    Matches CUDA kernel attention_value_computation_forward_v2 (case local_size=64):
        output[query_idx, head_idx, hdim_idx] =
            sum_l(attn_weight[query_idx, l, head_idx] * value_features[abs_key_idx_l, head_idx, hdim_idx])
    for valid (index_pair != -1) entries; 0 otherwise.

    Note: In the value computation kernel, the switch is on local_size (not hdim).
    case 64 means local_size=64, and shared memory stores 64 attn_weight values
    and 64 value indices per (query, head) block.

    In the CUDA kernel with local_size=64:
        - shared_attn_weight[64] caches attention weights per (query, head) block.
        - shared_value_indices[64] caches absolute key indices.
        - Weighted sum over local_size is computed with a loop.
        - Invalid positions (index_pair == -1) are skipped.

    Shapes:
        attn_weight:    [total_query_num, local_size, nhead]
        value_features: [total_key_num, nhead, hdim]
        index_pair:     [total_query_num, local_size]
        output:         [total_query_num, nhead, hdim]

    CUDA-to-PyTorch line mapping, in execution order:
        1. cu:32-37 block/thread ids and bounds -> py: tensor shapes.
        2. cu:40-44 key_start_idx -> py: _compute_key_offset + abs_key_idx.
        3. cu:48-61 shared_attn_weight/shared_value_indices fill -> py:
           valid_mask, safe_idx, torch.gather, and attn_weight tensor itself.
        4. cu:64 output pointer arithmetic -> py: native output layout.
        5. cu:66-72 local_size weighted-sum loop -> py: masked broadcast
           multiply + sum(dim=1).

    Intermediate variable correspondence:
        CUDA hdim_idx/head_idx/query_idx <-> PyTorch tensor axes [D, H, Q]
        CUDA shared_attn_weight[i]       <-> PyTorch attn_weight / masked_attn
        CUDA cur_key_idx                 <-> PyTorch abs_key_idx
        CUDA shared_value_indices[i]     <-> PyTorch safe_idx / valid_mask
        CUDA value_features[...]         <-> PyTorch gathered_values
        CUDA attn_result                 <-> PyTorch output before return
    """
    # cu:32-37 -> PyTorch names the CUDA block/thread axes as tensor dimensions.
    total_query_num, local_size = index_pair.size()
    nhead, hdim = value_features.shape[1], value_features.shape[2]

    # cu:40-44 -> vectorized key_start_idx and absolute value indices.
    key_offset = _compute_key_offset(key_batch_cnt, index_pair_batch)
    abs_key_idx = index_pair + key_offset.unsqueeze(-1)  # [total_query_num, local_size]
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

    return output


def _attention_value_computation_backward_pytorch(
        grad_out, query_batch_cnt, key_batch_cnt, index_pair_batch,
        index_pair, attn_weight, value_features):
    """
    PyTorch native backward for attention value computation.

    Matches CUDA kernel attention_value_computation_backward_v2 (case local_size=64):
        grad_attn_weight[query_idx, l, head_idx] =
            sum_i(grad_out[query_idx, head_idx, i] * value_features[abs_key_idx_l, head_idx, i])
            (accumulated via shared memory atomicAdd across hdim threads)
        grad_value_features[abs_key_idx, head_idx, i] +=
            grad_out[query_idx, head_idx, i] * attn_weight[query_idx, l, head_idx]
            (accumulated via global atomicAdd across queries referencing same value)

    Note: In the value computation backward kernel, the switch is on local_size (not hdim).
    case 64 means local_size=64, and shared memory stores 64 attn_weight values,
    64 grad_attn_weight accumulators, and 64 value indices.

    In the CUDA kernel with local_size=64:
        - shared_attn_weight[64] and shared_grad_attn_weight[64] per (query, head) block.
        - shared_value_indices[64] caches absolute key indices.
        - grad_attn_weight accumulated via shared memory atomicAdd across hdim threads.
        - grad_value_features uses global atomicAdd for concurrent access from different blocks.

    The scatter_add for grad_value_features matches CUDA atomicAdd behavior
    (multiple queries may reference the same value).

    Shapes:
        grad_out:            [total_query_num, nhead, hdim]
        attn_weight:         [total_query_num, local_size, nhead]
        value_features:      [total_key_num, nhead, hdim]
        grad_attn_weight:    [total_query_num, local_size, nhead]
        grad_value_features: [total_key_num, nhead, hdim]

    CUDA-to-PyTorch line mapping, in execution order:
        1. cu:126-131 block/thread ids and bounds -> py: tensor shapes.
        2. cu:134-138 key_start_idx -> py: _compute_key_offset + abs_key_idx.
        3. cu:142-156 shared_attn_weight/shared_grad_attn_weight/
           shared_value_indices fill -> py: masked_attn, grad_attn_weight,
           valid_mask, safe_idx, gathered_values.
        4. cu:159 gradient load -> py: grad_out.unsqueeze(1).
        5. cu:160-164 shared_grad_attn_weight atomicAdd -> py:
           (grad_out * gathered_values).sum(dim=-1) plus valid mask.
        6. cu:165-167 global grad_value_features atomicAdd -> py:
           grad_value_contrib + scatter_add_.
        7. cu:171-172 copy shared grad to global attn grad -> py:
           grad_attn_weight is already in output layout.

    Intermediate variable correspondence:
        CUDA gradient                        <-> PyTorch grad_out.unsqueeze(1)
        CUDA shared_attn_weight[i]           <-> PyTorch attn_weight / masked_attn
        CUDA shared_grad_attn_weight[i]      <-> PyTorch grad_attn_weight
        CUDA cur_key_idx                     <-> PyTorch abs_key_idx
        CUDA shared_value_indices[i]         <-> PyTorch safe_idx / valid_mask
        CUDA value_features[...]             <-> PyTorch gathered_values
        CUDA grad_value_features atomic target <-> PyTorch grad_value_features.scatter_add_
    """
    # cu:126-131 -> PyTorch names the CUDA block/thread axes as tensor dimensions.
    total_query_num, local_size = index_pair.size()
    nhead, hdim = value_features.shape[1], value_features.shape[2]
    total_key_num = value_features.shape[0]

    # cu:134-138 -> vectorized key_start_idx and absolute value indices.
    key_offset = _compute_key_offset(key_batch_cnt, index_pair_batch)
    abs_key_idx = index_pair + key_offset.unsqueeze(-1)  # [total_query_num, local_size]
    valid_mask = (index_pair != -1).to(grad_out.dtype)    # [total_query_num, local_size]
    safe_idx = abs_key_idx.clamp(min=0)

    # cu:149-155 and cu:164 -> shared_value_indices plus pointer arithmetic become gather.
    gather_idx = safe_idx.reshape(-1, 1, 1).expand(-1, nhead, hdim)
    gathered_values = torch.gather(value_features, 0, gather_idx).reshape(
        total_query_num, local_size, nhead, hdim)

    # cu:145-147 and cu:161 -> shared_attn_weight and invalid skip become masked_attn.
    masked_attn = attn_weight * valid_mask.unsqueeze(-1)  # [total_query_num, local_size, nhead]

    # cu:159-164 -> hdim threads atomically reduce grad_attn_weight; PyTorch sums dim=-1.
    grad_attn_weight = (grad_out.unsqueeze(1) * gathered_values).sum(dim=-1)
    # cu:160-161 -> invalid pairs do not update shared_grad_attn_weight.
    grad_attn_weight = grad_attn_weight * valid_mask.unsqueeze(-1)

    # cu:165-167 -> each valid pair contributes to a global value-gradient atomicAdd.
    grad_value_contrib = grad_out.unsqueeze(1) * masked_attn.unsqueeze(-1)

    grad_value_contrib_flat = grad_value_contrib.reshape(-1, nhead, hdim)
    safe_idx_flat = safe_idx.reshape(-1, 1, 1).expand(-1, nhead, hdim)

    # cu:165-167 -> global atomicAdd is represented by scatter_add_ on value index.
    grad_value_features = torch.zeros(
        total_key_num, nhead, hdim,
        dtype=value_features.dtype, device=value_features.device)
    grad_value_features.scatter_add_(0, safe_idx_flat, grad_value_contrib_flat)

    return grad_attn_weight, grad_value_features


# ============== Function classes with CUDA / PyTorch branching ==============


""" Attention computation code v2."""
class AttentionWeightComputation(Function):
    """
    Generate the attention weight matrix based on:
        * the generated attention pair index (total_query_num, local_size);
        * query features (total_query_num, nhead, hdim)
        * key features (total_key_num, nhead, hdim)
    Generate the attention weight matrix.
        * (total_query_num, local_size, nhead)
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

        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = key_features.size()

        # Need to ensure that every tensor in query features have an output.
        assert total_query_num == query_features.shape[0]

        backend = 'cuda' if _MTR_USE_CUDA else 'pytorch'
        dump_counter = _next_dump_counter('weight_forward') if _DUMP_DIR else None
        output = query_features.new_zeros(total_query_num, local_size, nhead)
        dump_data = {
            'backend': backend,
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
            _dump_attention_call('weight_forward', backend, 'before', dump_counter, dump_data)

        if _MTR_USE_CUDA:
            attention_cuda.attention_weight_computation_wrapper_v2(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                output)
        else:
            output = _attention_weight_computation_forward_pytorch(
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features)

        if dump_counter is not None:
            dump_data['phase'] = 'after'
            dump_data['output'] = output
            _dump_attention_call('weight_forward', backend, 'after', dump_counter, dump_data)

        ctx.for_backwards = (
            b, total_query_num, local_size, total_key_num, nhead, hdim,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, query_features, key_features
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
        (b, total_query_num, local_size, total_key_num, nhead, hdim,
         query_batch_cnt, key_batch_cnt, index_pair_batch,
         index_pair, query_features, key_features) = ctx.for_backwards

        if _MTR_USE_CUDA:
            grad_query_features = Variable(torch.cuda.FloatTensor(
                total_query_num, nhead, hdim).zero_())
            grad_key_features = Variable(torch.cuda.FloatTensor(
                total_key_num, nhead, hdim).zero_())

            grad_out_data = grad_out.data.contiguous()
            attention_cuda.attention_weight_computation_grad_wrapper_v2(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features,
                grad_out_data, grad_query_features.data, grad_key_features.data)
        else:
            grad_query_features, grad_key_features = _attention_weight_computation_backward_pytorch(
                grad_out.contiguous(), query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, query_features, key_features)

        return None, None, None, None, grad_query_features, grad_key_features


attention_weight_computation = AttentionWeightComputation.apply


class AttentionValueComputation(Function):
    """
    Generate the attention result based on:
        * the generated attention pair index (total_query_num, local_size);
        * value features (total_key_num, nhead, hdim)
        * attn_weight (total_query_num, local_size, nhead)
    Generate the attention result.
        * (total_query_num, nhead, hdim)
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

        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = value_features.size()

        # Need to ensure that every tensor in query features have an output.
        assert total_query_num == attn_weight.shape[0]

        backend = 'cuda' if _MTR_USE_CUDA else 'pytorch'
        dump_counter = _next_dump_counter('value_forward') if _DUMP_DIR else None
        output = value_features.new_zeros(total_query_num, nhead, hdim)
        dump_data = {
            'backend': backend,
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
            _dump_attention_call('value_forward', backend, 'before', dump_counter, dump_data)

        if _MTR_USE_CUDA:
            attention_cuda.attention_value_computation_wrapper_v2(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                output)
        else:
            output = _attention_value_computation_forward_pytorch(
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features)

        if dump_counter is not None:
            dump_data['phase'] = 'after'
            dump_data['output'] = output
            _dump_attention_call('value_forward', backend, 'after', dump_counter, dump_data)

        ctx.for_backwards = (
            b, total_query_num, local_size, total_key_num, nhead, hdim,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, attn_weight, value_features
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
        (b, total_query_num, local_size, total_key_num, nhead, hdim,
         query_batch_cnt, key_batch_cnt, index_pair_batch,
         index_pair, attn_weight, value_features) = ctx.for_backwards

        if _MTR_USE_CUDA:
            grad_attn_weight = Variable(torch.cuda.FloatTensor(
                total_query_num, local_size, nhead).zero_())
            grad_value_features = Variable(torch.cuda.FloatTensor(
                total_key_num, nhead, hdim).zero_())

            grad_out_data = grad_out.data.contiguous()
            attention_cuda.attention_value_computation_grad_wrapper_v2(
                b, total_query_num, local_size, total_key_num, nhead, hdim,
                query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features,
                grad_out_data, grad_attn_weight.data, grad_value_features.data)
        else:
            grad_attn_weight, grad_value_features = _attention_value_computation_backward_pytorch(
                grad_out.contiguous(), query_batch_cnt, key_batch_cnt, index_pair_batch,
                index_pair, attn_weight, value_features)

        return None, None, None, None, grad_attn_weight, grad_value_features


attention_value_computation = AttentionValueComputation.apply
