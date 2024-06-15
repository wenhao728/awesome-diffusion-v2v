#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/07 16:06:35
@Desc    :   
@Ref     :   
    https://github.com/openai/CLIP
    https://github.com/mlfoundations/open_clip
    https://huggingface.co/docs/transformers/model_doc/clip#clip
'''
import logging
from pathlib import Path

import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

class ClipConsistency(BaseEvaluator):
    pretrained_model_name = 'openai/clip-vit-large-patch14'

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
        self.preprocessor = CLIPImageProcessor.from_pretrained(pretrained_model_name)
        self.model = CLIPVisionModel.from_pretrained(pretrained_model_name)
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
            feature: torch.Tensor = self.model(pixel_values=frame).pooler_output
            feature = feature / torch.norm(feature, dim=-1, keepdim=True)

            if i > 0:
                sim = max(0, (feature @ former_feature.T).cpu().squeeze().item())
                similarity.append(sim)
            former_feature = feature
        return sum(similarity) / len(similarity)