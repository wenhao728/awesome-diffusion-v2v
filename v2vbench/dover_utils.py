#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/26 16:38:16
@Desc    :   
@Ref     :   https://github.com/VQAssessment/DOVER/tree/master
'''
import logging
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from third_party.dover.dataset import (
    UnifiedFrameSampler,
    spatial_temporal_view_decomposition,
)

logger = logging.getLogger(__name__)

class DoverPreprocessor:
    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])

    def __init__(
        self,
        sample_types: Dict[str, Dict[str, int]],
    ):
        self.sample_types = sample_types
        self.temporal_samplers = {}
        for stype, sopt in sample_types.items():
            if "t_frag" not in sopt:
                # resized temporal sampling for TQE in DOVER
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                # temporal sampling for AQE in DOVER
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

    def __call__(self, frames: List[Image.Image]):
        views, _ = spatial_temporal_view_decomposition(
            frames, self.sample_types, self.temporal_samplers
        )

        for k, v in views.items():
            v: torch.Tensor
            num_clips = self.sample_types[k].get("num_clips", 1)
            views[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2).reshape(
                v.shape[0], num_clips, -1, *v.shape[2:]).transpose(0, 1)
        return views


def fuse_results(results: list):
    logger.debug(f'Before fuse: {results}')
    means = [0.1107, 0.08285]
    stds = [0.07355, 0.03774]
    weights = [0.6104, 0.3896]
    x = (results[0] - means[0]) / stds[0] * weights[0] + (results[1] + means[1]) / stds[1] * weights[1]
    return 1 / (1 + np.exp(-x))  # sigmoid