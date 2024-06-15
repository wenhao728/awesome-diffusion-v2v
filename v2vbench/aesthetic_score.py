#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/26 15:54:42
@Desc    :   
@Ref     :   https://github.com/christophschuhmann/improved-aesthetic-predictor
'''
import logging
from pathlib import Path

import torch
from aesthetics_predictor import AestheticsPredictorV1, AestheticsPredictorV2Linear
from transformers import CLIPProcessor

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

class AestheticScore(BaseEvaluator):
    # pretrained_model_name = 'shunk031/aesthetics-predictor-v1-vit-large-patch14'  # v1
    pretrained_model_name = 'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE'  # v2

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
        logger.debug(f"Loading model {pretrained_model_name}")
        self.preprocessor = CLIPProcessor.from_pretrained(pretrained_model_name)
        # self.model = AestheticsPredictorV1.from_pretrained(pretrained_model_name)
        self.model = AestheticsPredictorV2Linear.from_pretrained(pretrained_model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"Model {self.model.__class__.__name__} loaded")

    def range(self):
        return 0, 10

    def preprocess(self, sample):
        video = sample['edit_video']
        frames = []
        for i, frame in enumerate(video):
            frames.append(self.preprocessor(images=frame, return_tensors='pt').pixel_values)

        return frames

    @torch.no_grad()
    def evaluate(self, frames) -> float:
        score = []
        for i, frame in enumerate(frames):
            if i == 0:
                logger.debug(f"Input shape: {frame.shape}")
            frame = frame.to(self.device)
            prediction = self.model(pixel_values=frame).logits
            if i == 0:
                logger.debug(f"Output shape: {prediction.shape}")
            score.append(prediction.squeeze().cpu().item())
        return sum(score) / len(score)