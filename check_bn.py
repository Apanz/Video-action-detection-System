#!/usr/bin/env python3
import torch

checkpoint = torch.load('outputs/checkpoints/ucf101_best.pth', map_location='cpu')
state_dict = checkpoint['model_state_dict']

bn_keys = [k for k in state_dict.keys() if 'running_mean' in k or 'running_var' in k]
print(f'BatchNorm stats count: {len(bn_keys)}')

print('\nFirst 10 BatchNorm stats:')
for k in bn_keys[:10]:
    v = state_dict[k]
    has_nan = torch.isnan(v).any().item()
    has_inf = torch.isinf(v).any().item()
    print(f'{k}:')
    print(f'  shape={v.shape}, nan={has_nan}, inf={has_inf}')
    print(f'  range=[{v.min():.4f}, {v.max():.4f}]')
    print(f'  mean={v.mean():.6f}, std={v.std():.6f}')
    print()
