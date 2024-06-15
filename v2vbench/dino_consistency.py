#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/07 16:06:35
@Desc    :   
@Ref     :   
    https://github.com/facebookresearch/dinov2
    https://huggingface.co/docs/transformers/model_doc/dinov2#dinov2
'''
import logging
from pathlib import Path

import torch
from transformers import BitImageProcessor, Dinov2Model

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

class DinoConsistency(BaseEvaluator):
    pretrained_model_name = 'facebook/dinov2-base'

    def __init__(
        self,
        index_file: Path, 
        edit_video_dir: Path,
        reference_video_dir: Path,
        device: torch.device,
        pretrained_model_name: str = None,
    ):
        super().__init__(index_file, edit_video_dir, None, device)
        pretrained_model_name = pretrained_model_name or self.pretrained_model_name
        logger.debug(f"Loding model {pretrained_model_name}")
        self.preprocessor = BitImageProcessor.from_pretrained(pretrained_model_name)
        self.model = Dinov2Model.from_pretrained(pretrained_model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"Model {self.model.__class__.__name__} loaded")

    def range(self):
        return 0, 1

    def preprocess(self, sample):
        video = sample['edit_video']
        frames = []
        for i, frame in enumerate(video):
            frames.append(self.preprocessor(frame, return_tensors='pt').pixel_values)
        return frames

    @torch.no_grad()
    def evaluate(self, frames) -> float:
        similarity = []
        former_feature = None
        for i, frame in enumerate(frames):
            frame = frame.to(self.device)
            # pooled output (first image token)
            feature = self.model(pixel_values=frame).pooler_output
            feature: torch.Tensor = feature / torch.norm(feature, dim=-1, keepdim=True)

            if i > 0:
                sim = max(0, (feature @ former_feature.T).cpu().squeeze().item())
                similarity.append(sim)
            former_feature = feature
        return sum(similarity) / len(similarity)