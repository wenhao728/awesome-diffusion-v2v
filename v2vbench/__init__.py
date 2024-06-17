#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/07 13:48:51
@Desc    :   
@Ref     :   
'''
import gc
import importlib
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

cur_dir = Path('..').resolve().absolute()
if str(cur_dir) not in sys.path:
    sys.path.append(str(cur_dir))


class EvaluatorWrapper:
    _quality_metrics = ['aesthetic_score', 'clip_consistency', 'dino_consistency', 'dover_score']
    _alignment_metrics = ['clip_text_alignment', 'pick_score', 'viclip_text_alignment', 'motion_alignment']
    all_metrics = _quality_metrics + _alignment_metrics

    def __init__(
        self,
        device: Optional[torch.device] = None,
        metrics: Union[str, List[str]] = 'all',
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = metrics

    def _check_args(self, edit_video, reference_video, index_file, edit_prompt):
        if isinstance(self.metrics, str):
            if self.metrics == 'all':
                self.metrics = self.all_metrics
            else:
                self.metrics = self.metrics.split(',')
        not_supported_metrics = set(self.metrics) - set(self.all_metrics)
        self.metrics = list(set(self.metrics) - not_supported_metrics)
        if not_supported_metrics:
            warnings.warn(f'Unsupported metrics: {not_supported_metrics}')
        if self.metrics:
            logger.info(f'Using metrics: {self.metrics}')
        else:
            raise ValueError('No supported metrics provided')

        if not edit_video.exists():
            raise FileNotFoundError(f'Edit video: {edit_video} does not exist')
        if reference_video and not reference_video.exists():
            raise FileNotFoundError(f'Reference video: {reference_video} does not exist')
        if index_file:
            if not index_file.exists():
                raise FileNotFoundError(f'Index file: {index_file} does not exist')
        else:
            if not edit_prompt:
                raise ValueError(
                    'Edit prompt (for single video evaluation) OR index file (for batch evaluation) is required')

    def merge_results(self, results: Dict[str, Dict[str, List]]) -> pd.DataFrame:
        merged_results_df = None
        for metric, result in results.items():
            result_df = pd.DataFrame(result)
            result_df.rename(columns={'score': metric}, inplace=True)
            if merged_results_df is None:
                merged_results_df = result_df
            else:
                merged_results_df = pd.merge(merged_results_df, result_df, on=['video_id', 'edit_id'])
        return merged_results_df

    def evaluate(
        self,
        edit_video: Path,
        reference_video: Optional[Path] = None, 
        index_file: Optional[Path] = None,
        edit_prompt: Optional[str] = None,
        output_dir: Optional[Path] = None,
        save_results: bool = True,
        save_summary: bool = True,
        evaluator_kwargs: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Dict[str, List]]:
        """Evaluate the editing video(s) using the specified metrics

        Args:
            edit_video (Path): Path to the editing video (single video) or directory containing editing videos (batch).
            reference_video (Optional[Path], optional): Path to the reference video (single video) or directory 
                containing reference videos (batch). Defaults to None.
            index_file (Optional[Path], optional): Index file containing metadata for batch evaluation. Defaults to 
                None.
            edit_prompt (Optional[str], optional): Edit prompt for single video evaluation. Defaults to None.
            output_dir (Optional[Path], optional): Directory to save the results and summary files. Defaults to None.
            save_results (bool, optional): Save the results in a CSV file. Only applicable when output_dir is provided.
                Defaults to True.
            save_summary (bool, optional): Save the summary in a CSV file. Only applicable when output_dir is provided.
                Defaults to True.
            evaluator_kwargs (Optional[Dict[str, Dict]], optional): Additional keyword arguments for the evaluators.
                Defaults to None.

        Returns:
            Dict[str, Dict[str, List]]: Results for each metric for each sample, formatted as below:
                {
                    'metric1_name': {
                        'video_id': [video_id1, video_id2, ...],
                        'edit_id: [edit_id1, edit_id2, ...],
                        'score': [score1, score2, ...]
                    },
                    ...
                }
        """
        reference_video = Path(reference_video).resolve() if reference_video else None
        edit_video = Path(edit_video).resolve()
        index_file = Path(index_file).resolve() if index_file else None
        edit_prompt = edit_prompt
        self._check_args(edit_video, reference_video, index_file, edit_prompt)

        # evaluate results for each sample
        results = {}
        for metric in tqdm(self.metrics):
            try:
                module = importlib.import_module(f'.{metric}', package='v2vbench')
                evaluator_cls_name = ''.join([w[0].upper() + w[1:] for w in metric.split('_')])
                evaluator_cls = getattr(module, evaluator_cls_name)
                kwargs = evaluator_kwargs[metric] if evaluator_kwargs and metric in evaluator_kwargs else {}

                evaluator = evaluator_cls(
                    index_file=index_file,
                    edit_video_dir=edit_video,
                    reference_video_dir=reference_video,
                    edit_prompt=edit_prompt,
                    device=self.device,
                    **kwargs,
                )
            except Exception as e:
                raise NotImplementedError(f'Failed to initalize {metric} metrics: {e}')

            result = evaluator()
            results[metric] = result
            
            # release memory after evaluating each metric
            del evaluator
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # merge the results into a single dataframe
        merged_results_df = self.merge_results(results)
        # aggregate into a single-row summary
        summary_df = merged_results_df[self.metrics].mean(axis=0).to_frame().T
        logger.info(reference_video.name)
        logger.info(summary_df)
        
        if output_dir is not None and (save_results or save_summary):
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            if save_results:
                merged_results_df.to_csv(output_dir / 'results.csv', index=False)
                logger.info(f'Results saved at: {output_dir / "results.csv"}')
            if save_summary:
                summary_df.to_csv(output_dir / 'summary.csv', index=False)
                logger.info(f'Summary saved at: {output_dir / "summary.csv"}')

        return results