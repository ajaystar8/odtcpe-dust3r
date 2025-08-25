"""
# TODO: UNUSED SCRIPT: consider removing it in the future.

This file is derived from [DUSt3R](https://github.com/ajaystar8/odtcpe-dust3r/blob/main/dust3r/inference.py).
Modified for [ODTCPE] by Ajay Rajendra Kumar

Original header:
"""
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
from dust3r.dust3r.utils.device import to_cpu, collate_with_cat


def forward_pass(batch, model, device, use_amp=False, ret=None):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    with torch.amp.autocast('cuda', enabled=bool(use_amp)):
        feat1, feat2 = model(view1, view2)

    result = dict(view1=view1, view2=view2, feat1=feat1, feat2=feat2)
    return result[ret] if ret else result

@torch.no_grad()
def get_feature_vectors(pairs, model, device, batch_size=8, verbose=False):
    feature_vectors = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = forward_pass(collate_with_cat(pairs[i:i + batch_size]), model, device)
        feature_vectors.append(to_cpu(res))

    feature_vectors = collate_with_cat(feature_vectors, lists=multiple_shapes)
    return feature_vectors


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)