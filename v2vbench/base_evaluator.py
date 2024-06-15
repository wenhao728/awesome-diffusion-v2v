#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/26 13:48:08
@Desc    :   
@Ref     :   
'''
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Union

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from .utils import load_video

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    def __init__(
        self, 
        index_file: Path, 
        edit_video_dir: Path,
        reference_video_dir: Path,
        device: torch.device,
        stride: int = 1,
    ):
        self.index_file = index_file
        self.edit_video_dir = edit_video_dir
        # self.method = self.edit_video_dir.stem
        self.reference_video_dir = reference_video_dir
        self.device = device
        # stride for loading video frames, default is 1 - load all frames
        # if stride is 2, load every other frame, if stride is 3, load every third frame, etc.
        self.stride = stride

    @property
    @abstractmethod
    def range(self) -> Tuple[Union[float, int]]:
        """
        Returns:
            Tuple[Union[float, int]]: the worse and best value of the metric
        """
        raise NotImplementedError
    
    @property
    def direction(self) -> str:
        return 'maximize' if self.range[0] < self.range[1] else 'minimize'

    def _data_iter(self):
        config: OmegaConf = OmegaConf.load(self.index_file)
        for sample in config['data']:
            # reference video is optional
            reference_video_path = None
            reference_video = None

            if self.reference_video_dir is not None:
                reference_video_path = self.reference_video_dir / sample['video_id']
                reference_video = load_video(reference_video_path, stride=self.stride)

            # iterate over edit
            for edit_id, edit in enumerate(sample['edit']):
                edit_video_path = self.edit_video_dir / sample['video_id'] / str(edit_id)
                edit_video = load_video(edit_video_path, stride=self.stride)
                yield {
                    'video_id': sample['video_id'],
                    'prompt': sample['prompt'],
                    'reference_video': reference_video,
                    'reference_video_path': reference_video_path if reference_video is not None else None,
                    'edit_id': edit_id,
                    'edit_type': edit['type'],
                    'edit_prompt': edit['prompt'],
                    'edit_video': edit_video,
                    'edit_video_path': edit_video_path,
                }

    def preprocess(self, sample):
        return sample

    @abstractmethod
    def evaluate(self, sample) -> float:
        raise NotImplementedError

    def __call__(self):
        results = defaultdict(list)
        for sample in tqdm(self._data_iter()):
            score = self.evaluate(self.preprocess(sample))

            # results['method'].append(self.method)
            results['video_id'].append(sample['video_id'])
            results['edit_id'].append(sample['edit_id'])
            results['score'].append(score)
        return results