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

import numpy as np
import torch
from transformers import BitImageProcessor, Dinov2Model

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

class DinoImageAlignment(BaseEvaluator):
    pretrained_model_name = 'facebook/dinov2-base'

    def __init__(
        self,
        index_file: Path, 
        edit_video_dir: Path,
        reference_video_dir: Path,
        edit_prompt: str,
        device: torch.device,
        pretrained_model_name: str = None,
    ):
        super().__init__(index_file, edit_video_dir, None, edit_prompt, device)
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
        reference_image = self.preprocessor(sample['reference_image'], return_tensors='pt').pixel_values
        return frames, reference_image

    @torch.no_grad()
    def evaluate(self, args) -> float:
        frames, reference_image = args
        similarity = []
        conformity = []

        reference_image = reference_image.to(self.device)
        reference_feature = self.model(pixel_values=reference_image).pooler_output
        reference_feature: torch.Tensor = reference_feature / torch.norm(reference_feature, dim=-1, keepdim=True)

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

            conformity.append(max(0, (feature @ reference_feature.T).cpu().squeeze().item()))

        max_weight = 0.4
        mean_weight = 0.3
        min_weight = 0.3
        return float(
            max_weight * np.max(conformity) +
            mean_weight * np.mean(similarity) +
            min_weight * np.min(similarity)
        )