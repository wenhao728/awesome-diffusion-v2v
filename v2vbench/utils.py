#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/07 14:49:35
@Desc    :   
@Ref     :   
'''
import logging
from pathlib import Path
from typing import List

import decord
from PIL import Image, ImageSequence

decord.bridge.set_bridge('native')
logger = logging.getLogger(__name__)
_supported_image_suffix = ['.jpg', '.jpeg', '.png']
_supported_video_suffix = ['.mp4', '.gif']


def _load_video_from_image_dir(video_dir: Path) -> List[Image.Image]:
    logger.debug(f'Loading video from image directory: {video_dir}')
    frames = []
    for file in sorted(video_dir.iterdir()):
        if file.suffix not in _supported_image_suffix:
            logger.debug(f'Skipping file: {file}')
            continue
        frame = Image.open(file).convert('RGB')
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f'No image found in {video_dir}, supported image suffix: {_supported_image_suffix}')
    
    return frames


def _load_video_from_video_file(video_file: Path) -> List[Image.Image]:
    logger.debug(f'Loading video from video file: {video_file}')
    if video_file.suffix == '.mp4':
        video_reader = decord.VideoReader(str(video_file), num_threads=1)
        frames = []
        for i in range(len(video_reader)):
            frames.append(Image.fromarray(video_reader[i].asnumpy()))
        return frames

    elif video_file.suffix == '.gif':
        frames = []
        for f in ImageSequence.Iterator(Image.open(video_file)):
            frame = f.convert('RGB')
            frames.append(frame)
        return frames

    else:
        raise NotImplementedError(
            f'Unsupported video file: {video_file}, supported suffix: {_supported_video_suffix}')


def load_video(video_file_or_dir: Path, start_frame: int = 0, stride: int = 1) -> List[Image.Image]:
    """
    Args:
        video_file_or_dir (Path): path to video file or directory containing images
        start_frame (int): start frame index
        stride (int): stride for frame sampling
    Returns:
        List[Image.Image]: list of frames, RGB
    """
    if not video_file_or_dir.exists():
        logger.debug(f'Video file or directory does not exist: {video_file_or_dir}, trying to find alternative')
        for suffix in _supported_video_suffix:
            if (video_file_or_dir.with_suffix(suffix)).exists():
                video_file_or_dir = video_file_or_dir.with_suffix(suffix)
                logger.debug(f'Found video file: {video_file_or_dir}')
                break
        else:
            raise FileNotFoundError(f'Reference video: {video_file_or_dir} does not exist')

    if video_file_or_dir.is_dir():
        buffer = _load_video_from_image_dir(video_file_or_dir)
    elif video_file_or_dir.is_file():
        buffer = _load_video_from_video_file(video_file_or_dir)
    else:
        # should not reach here
        raise NotImplementedError(f'{video_file_or_dir} is not a valid file or directory')
    
    video = buffer[start_frame::stride]
    logger.debug(f'Raw video frames: {len(buffer)}, sampled video frames: {len(video)}')
    logger.debug(f'Frame size: {video[0].size}')
    return video