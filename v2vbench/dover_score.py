#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/26 16:31:28
@Desc    :   
@Ref     :   https://github.com/VQAssessment/DOVER/tree/master
'''
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from third_party.dover.models import DOVER

from .base_evaluator import BaseEvaluator
from .dover_utils import DoverPreprocessor, fuse_results

logger = logging.getLogger(__name__)

class DoverScore(BaseEvaluator):
    pretrained_config_file = (Path(__file__) / '../../third_party/dover/config.yaml').resolve().absolute()
    pretrained_checkpoint = (Path(__file__) / '../../checkpoints/DOVER/DOVER.pth').resolve().absolute()

    def __init__(
        self,
        index_file: Path, 
        edit_video_dir: Path,
        reference_video_dir: Path,
        edit_prompt: str,
        device: torch.device,
        pretrained_config_file: str = None,
        pretrained_checkpoint: str = None,
    ):
        super().__init__(index_file, edit_video_dir, None, edit_prompt, device)

        pretrained_config_file = pretrained_config_file or self.pretrained_config_file
        pretrained_checkpoint = pretrained_checkpoint or self.pretrained_checkpoint

        logger.debug(f"Loding model {pretrained_checkpoint}")
        config = OmegaConf.to_container(OmegaConf.load(pretrained_config_file))
        self.preprocessor = DoverPreprocessor(config["data"])
        self.model = DOVER(**config['model'])
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(pretrained_checkpoint, map_location=self.device))
        self.model.eval()
        logger.debug(f"Model {self.model.__class__.__name__} loaded")

    def range(self):
        return 0, 1

    def preprocess(self, sample):
        video = sample['edit_video']

        views = self.preprocessor(video)
        for k, v in views.items():
            views[k] = v.to(self.device)
        return views
 
    @torch.no_grad()
    def evaluate(self, views) -> float:
        results = [r.mean().item() for r in self.model(views)]
        # score
        scores = fuse_results(results)
        return scores