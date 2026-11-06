"""Experimental MMDiT TP2 and PP2 adapters, outside production inference.

TP2 shards attention heads and MLP channels in the main double/single blocks.
Encoders, conditioning refiner, normalizations and modulation are replicated.
PP2 partitions the existing blocks between devices; one clip has no microbatch
pipeline overlap. Neither adapter divides the temporal sequence or changes SDPA.
Call only on a newly loaded, evaluated model inside an inference/no-grad scope.
"""
from dataclasses import fields, is_dataclass, replace
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F


def _require_model(model):
    if model.training:
        raise ValueError('Parallel experiments require an evaluated inference model.')
    if type(model).__name__ != 'HunyuanMotionMMDiT':
        raise ValueError('This adapter supports the pinned FlowHMR HunyuanMotionMMDiT only.')
    if getattr(model, '_experimental_parallel', None):
        raise ValueError('Load a fresh model before applying another parallel mode.')
    if model.attention_backend != 'query_batched':
        raise ValueError('Keep the validated query-batched attention backend.')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Keep the validated FP32 generator precision.')


class ColumnShard(nn.Module):
    def __init__(self, source, rows):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().index_select(0, rows).clone(), requires_grad=False)
        self.bias = None if source.bias is None else nn.Parameter(source.bias.detach().index_select(0, rows).clone(), requires_grad=False)
        self.in_features = source.in_features
        self.out_features = len(rows)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowShard(nn.Module):
    def __init__(self, source, columns, group):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().index_select(1, columns).clone(), requires_grad=False)
        self.bias = None if source.bias is None else nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        self.in_features = len(columns)
        self.out_features = source.out_features
        self.group = group

    def forward(self, x):
        if self.training or torch.is_grad_enabled() or torch.is_autocast_enabled():
            raise RuntimeError('Tensor parallel reductions support inference only.')
        result = F.linear(x, self.weight, None)
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.group)
        # Replicated bias is added once, after the sum of partial dot products.
        return result if self.bias is None else result + self.bias


def _slice(size, rank, world, device):
    if size % world:
        raise ValueError(f'Cannot evenly shard width {size} across {world} ranks.')
    return torch.arange(rank * (size // world), (rank + 1) * (size // world), device=device)


def tensor_parallel(model, group=None):
    """Mutate a fresh model into two attention/MLP shards; each rank gets full inputs."""
    _require_model(model)
    if not dist.is_initialized() or dist.get_world_size(group) != 2:
        raise ValueError('TP2 requires an initialized two-rank process group.')
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    before = sum(p.numel() for p in model.parameters())
    for block in model.double_blocks:
        if block.num_heads % world:
            raise ValueError('The attention head count must divide evenly between ranks.')
        width = block.feat_dim
        dev = block.motion_qkv.weight.device
        heads = _slice(width, rank, world, dev)
        qkv = torch.cat([heads + k * width for k in range(3)])
        for stream in ('motion', 'text'):
            name = stream + '_qkv'
            setattr(block, name, ColumnShard(getattr(block, name), qkv))
            name = stream + '_out_proj'
            setattr(block, name, RowShard(getattr(block, name), heads, group))
            mlp = getattr(block, stream + '_mlp')
            if not isinstance(mlp.norm, nn.Identity):
                raise ValueError('MLP channel normalization needs a separate distributed implementation.')
            channels = _slice(mlp.fc1.out_features, rank, world, dev)
            mlp.fc1 = ColumnShard(mlp.fc1, channels)
            mlp.fc2 = RowShard(mlp.fc2, channels, group)
        block.num_heads //= world
    for block in model.single_blocks:
        if block.num_heads % world:
            raise ValueError('The attention head count must divide evenly between ranks.')
        width, hidden = block.feat_dim, block.mlp_hidden_dim
        dev = block.linear1.weight.device
        heads = _slice(width, rank, world, dev)
        channels = _slice(hidden, rank, world, dev)
        rows = torch.cat([heads + k * width for k in range(3)] + [channels + 3 * width])
        columns = torch.cat([heads, channels + width])
        block.linear1 = ColumnShard(block.linear1, rows)
        block.linear2 = RowShard(block.linear2, columns, group)
        # These fields only describe the local fused projection/reshape in forward.
        # LayerNorm, residuals and modulation keep their full original dimensions.
        block.feat_dim //= world
        block.mlp_hidden_dim //= world
        block.num_heads //= world
    model._experimental_parallel = dict(mode='tp2', rank=rank, world_size=world,
        parameters_before=before, parameters_local=sum(p.numel() for p in model.parameters()),
        replicated='input/condition encoders, token refiner, norms, modulation, final head')
    return model.eval()


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _move(getattr(value, field.name), device) for field in fields(value)})
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    return value


class _Transfers:
    def __init__(self):
        self.cache = {}

    def clear(self, *unused):
        self.cache.clear()

    def static(self, value, device):
        key = (id(value), str(device))
        item = self.cache.get(key)
        if item is None or item[0] is not value:
            item = (value, _move(value, device))
            self.cache[key] = item
        return item[1]


class DeviceBlock(nn.Module):
    def __init__(self, inner, device, transfers, return_device=None):
        super().__init__()
        self.inner = inner.to(device)
        self.device = torch.device(device)
        self.transfers = transfers
        self.return_device = return_device

    def forward(self, *args, **kwargs):
        if self.training or torch.is_grad_enabled() or torch.is_autocast_enabled():
            raise RuntimeError('Pipeline experiments support inference only.')
        if kwargs.get('return_attn') or kwargs.get('attn_collector') is not None:
            raise ValueError('Attention-map collection is outside this inference experiment.')
        args = tuple(_move(value, self.device) for value in args)
        static = {'adapter', 'attn_mask', 'attention_plan'}
        kwargs = {name: (self.transfers.static(value, self.device) if name in static else _move(value, self.device))
                  for name, value in kwargs.items()}
        output = self.inner(*args, **kwargs)
        return output if self.return_device is None else _move(output, self.return_device)


def pipeline_parallel(model, devices=('cuda:0', 'cuda:1'), split=None):
    """Place contiguous main-network blocks on two devices, preserving the full sequence."""
    _require_model(model)
    if len(devices) != 2:
        raise ValueError('PP2 needs exactly two devices.')
    devices = tuple(map(torch.device, devices))
    if devices[0].type == 'cuda' and devices[0] == devices[1]:
        raise ValueError('PP2 needs distinct CUDA devices.')
    blocks = list(model.double_blocks) + list(model.single_blocks)
    counts = [sum(p.numel() for p in b.parameters()) for b in blocks]
    if split is None:
        split = min(range(1, len(blocks)), key=lambda k: abs(sum(counts[:k]) - sum(counts) / 2))
    if not 0 < split < len(blocks):
        raise ValueError('The pipeline split must leave blocks on both devices.')
    model.to(devices[0])
    transfers = _Transfers()
    for index, block in enumerate(blocks):
        device = devices[0] if index < split else devices[1]
        # Return the last full hidden state to the original device before the
        # upstream long-skip and final-head logic. No sequence/key is removed.
        wrapped = DeviceBlock(block, device, transfers, devices[0] if index == len(blocks) - 1 else None)
        if index < len(model.double_blocks):
            model.double_blocks[index] = wrapped
        else:
            model.single_blocks[index - len(model.double_blocks)] = wrapped
    model.register_forward_pre_hook(transfers.clear)
    model.register_forward_hook(lambda *unused: transfers.clear(), always_call=True)
    model._experimental_parallel = dict(mode='pp2', devices=list(map(str, devices)), split=split,
        blocks=len(blocks), block_parameters_per_stage=[sum(counts[:split]), sum(counts[split:])],
        overlap=False, note='One full-sequence clip; no microbatch overlap.')
    return model.eval()
