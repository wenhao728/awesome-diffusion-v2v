#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/27 12:30:53
@Desc    :   
@Ref     :   
'''
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

from third_party.gmflow.models import GMFlow
from third_party.gmflow.utils import InputPadder, write_flow

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

class MotionAlignment(BaseEvaluator):
    pretrained_config_file = (
        Path(__file__) / '../../third_party/gmflow/config.yaml').resolve().absolute()
    pretrained_checkpoint = (
        Path(__file__) / '../../checkpoints/gmflow/gmflow_sintel-0c07dcb3.pth').resolve().absolute()

    def __init__(
        self,
        index_file: Path, 
        edit_video_dir: Path,
        reference_video_dir: Path,
        edit_prompt: str,
        device: torch.device,
        pretrained_config_file: str = None,
        pretrained_checkpoint: str = None,
        cache_flow: bool = False,
    ):
        super().__init__(index_file, edit_video_dir, reference_video_dir, edit_prompt, device)
        pretrained_config_file = pretrained_config_file or self.pretrained_config_file
        pretrained_checkpoint = pretrained_checkpoint or self.pretrained_checkpoint

        logger.debug(f"Loding model {pretrained_checkpoint}")
        config = OmegaConf.to_container(OmegaConf.load(pretrained_config_file))
        self.padder = InputPadder(**config['data'])
        self.model = GMFlow(**config['model'])
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(pretrained_checkpoint, map_location=self.device)['model'])
        self.model.eval()
        logger.debug(f"Model {self.model.__class__.__name__} loaded")

        self.cache_flow = cache_flow

    def range(self):
        return -1 * np.inf, 0

    def get_flow_file(self, reference_video_path: Path):
        if reference_video_path is None:
            return Path('')

        video_id = reference_video_path.stem
        flow_dir = reference_video_path.parent.parent / 'flow'
        return flow_dir / f'{video_id}.pt'

    def preprocess(self, sample):
        # load edit frames
        edit_frames = []
        for i, frame in enumerate(sample['edit_video']):
            frame = torch.from_numpy(np.array(frame).astype(np.uint8)).permute(2, 0, 1).float().to(self.device)
            frame = self.padder.pad(frame)[0][None]  # (B, C, H, W)
            edit_frames.append(frame)

        reference_frames = []
        reference_flow = None
        reference_flow_file = self.get_flow_file(sample['reference_video_path'])

        # try to load cached flow, only for batch evaluation
        try:
            logger.debug(f"Loading cached flow from {reference_flow_file}")
            reference_flow = torch.load(reference_flow_file, map_location=self.device)
        # cached flow not found, compute flow
        except FileNotFoundError or AttributeError:
            reference_flow_file = None
            logger.debug(f"Flow file not found, loading reference frames.")
            for i, frame in enumerate(sample['reference_video']):
                frame = torch.from_numpy(np.array(frame).astype(np.uint8)).permute(2, 0, 1).float().to(self.device)
                frame = self.padder.pad(frame)[0][None]  # (B, C, H, W)
                reference_frames.append(frame)

        return edit_frames, reference_frames, reference_flow_file, reference_flow

    @torch.no_grad()
    def extract_flow(self, frames: List[torch.Tensor]):
        flows = []

        for i, frame in enumerate(frames):
            if i > 0:
                flow_pr = self.model(
                    prev_frame, frame,
                    attn_splits_list=[2],
                    corr_radius_list=[-1],
                    prop_radius_list=[-1],
                )['flow_preds'][-1][0]  # (2, H, W)
                flows.append(flow_pr)
                if i == 1:
                    logger.debug(f"Flow shape: {flow_pr.shape}")
            prev_frame = frame

        return torch.stack(flows, dim=0)  # (T-1, 2, H, W)

    @torch.no_grad()
    def evaluate(self, args) -> float:
        edit_frames, reference_frames, reference_flow_file, reference_flow = args
        edit_flow = self.extract_flow(edit_frames)

        # calculate flow for reference video if not cached
        if reference_flow_file is None:
            reference_flow = self.extract_flow(reference_frames)
            if self.cache_flow:
                write_flow(reference_flow, reference_flow_file)
                logger.debug(f"Flow saved to {reference_flow_file}")

        # calculate alignment score: EPE
        score = torch.sum((edit_flow - reference_flow) ** 2, dim=1).sqrt().mean().cpu().item()
        return - score
