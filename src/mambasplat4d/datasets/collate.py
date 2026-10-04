"""Collate functions for Gaussian batches."""
from typing import List, Dict, Any

import torch

# Collate only keys present in batch.
_GAUSSIAN_KEYS = {'position', 'quaternion', 'scale', 'opacity', 'sh_dc'}


def collate_contrastive_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dual-view contrastive collate.

    Item: {'view1': {gaussian keys, mask}, 'view2': {...}, 'label', 'class_name', 'object_id'}.
    Output packs [view1_0..view1_B-1, view2_0..view2_B-1] (2B).
    """
    B = len(batch)
    result: Dict[str, Any] = {}

    tensor_keys = [k for k in _GAUSSIAN_KEYS if k in batch[0]['view1']]
    for key in tensor_keys:
        v1 = torch.stack([item['view1'][key] for item in batch], dim=0)
        v2 = torch.stack([item['view2'][key] for item in batch], dim=0)
        result[key] = torch.cat([v1, v2], dim=0)

    m1 = torch.stack([item['view1']['mask'] for item in batch], dim=0)
    m2 = torch.stack([item['view2']['mask'] for item in batch], dim=0)
    result['mask'] = torch.cat([m1, m2], dim=0)

    # Same object's two views share pair id.
    result['pair_ids'] = torch.arange(B, dtype=torch.long).repeat(2)
    result['label'] = torch.tensor([item.get('label', -1) for item in batch], dtype=torch.long)
    result['class_name'] = [item.get('class_name', '') for item in batch]
    result['object_id'] = [item.get('object_id', '') for item in batch]
    return result


def collate_gaussians(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Static Gaussian collate: stack pre-padded tensors for present keys."""
    result = {}
    for key in _GAUSSIAN_KEYS:
        if key in batch[0]:
            result[key] = torch.stack([b[key] for b in batch])

    result['mask'] = torch.stack([b['mask'] for b in batch])
    result['label'] = torch.tensor([b['label'] for b in batch])
    result['class_name'] = [b['class_name'] for b in batch]
    result['sample_name'] = [b.get('sample_name', b.get('subject', '')) for b in batch]
    result['num_gaussians'] = [b.get('num_gaussians', 0) for b in batch]
    return result


def collate_sequences(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sequence collate: pad (T, N, ...) items to max T, return (B, T, N, ...)."""
    tensor_keys = [k for k in _GAUSSIAN_KEYS if k in batch[0]]

    max_T = max(b['position'].shape[0] for b in batch)
    N = batch[0]['position'].shape[1]
    B = len(batch)

    result = {}
    for key in tensor_keys:
        feat_shape = batch[0][key].shape[2:]  # (N, ...) -> (...)
        padded = torch.zeros(B, max_T, N, *feat_shape)
        for i, b in enumerate(batch):
            T_i = b[key].shape[0]
            padded[i, :T_i] = b[key]
        result[key] = padded

    # (B, T, N), True where valid
    mask = torch.zeros(B, max_T, N, dtype=torch.bool)
    for i, b in enumerate(batch):
        T_i = b['mask'].shape[0]
        mask[i, :T_i] = b['mask']
    result['mask'] = mask

    # (B, T), True for valid frames
    temporal_mask = torch.zeros(B, max_T, dtype=torch.bool)
    for i, b in enumerate(batch):
        T_i = b['num_frames']
        temporal_mask[i, :T_i] = True
    result['temporal_mask'] = temporal_mask

    result['label'] = torch.tensor([b['label'] for b in batch])
    result['class_name'] = [b['class_name'] for b in batch]
    result['subject'] = [b.get('subject', '') for b in batch]
    result['num_frames'] = [b['num_frames'] for b in batch]

    if 'video_idx' in batch[0]:
        result['video_idx'] = torch.tensor([b['video_idx'] for b in batch])

    return result
