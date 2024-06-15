#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/26 18:58:20
@Desc    :   
@Ref     :   
    https://github.com/openai/CLIP
    https://github.com/mlfoundations/open_clip
    https://huggingface.co/docs/transformers/model_doc/clip#clip
'''
import logging
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

class ClipTextAlignment(BaseEvaluator):
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
        self.preprocessor = CLIPProcessor.from_pretrained(pretrained_model_name)
        self.model = CLIPModel.from_pretrained(pretrained_model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"Model {self.model.__class__.__name__} loaded")

    def range(self):
        return 0, 100

    def preprocess(self, sample):
        text = sample['edit_prompt']
        video = sample['edit_video']
        
        text_inputs = self.preprocessor(
            text=text, padding=True, truncation=True, max_length=77, return_tensors='pt').to(self.device)
        
        image_inputs = []
        for frame in video:
            image_inputs.append(self.preprocessor(
                images=frame, padding=True, truncation=True, max_length=77, return_tensors='pt').to(self.device))
        
        return text_inputs, image_inputs

    @torch.no_grad()
    def evaluate(self, args) -> float:
        text_inputs, image_inputs = args
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = []
        for image_input in image_inputs:
            image_embs = self.model.get_image_features(**image_input)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            score = (self.model.logit_scale.exp() * (text_embs @ image_embs.T)).cpu().squeeze().item()
            scores.append(score)

        return sum(scores) / len(scores)