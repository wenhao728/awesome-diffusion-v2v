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
        index_file: Path, 
        edit_video_dir: Path,
        reference_video_dir: Optional[Path] = None, 
        device: Optional[torch.device] = None,
        metrics: Union[str, List[str]] = 'all',
    ):
        self.index_file = Path(index_file).resolve()
        self.edit_video_dir = Path(edit_video_dir).resolve()
        self.reference_video_dir = Path(reference_video_dir).resolve() if reference_video_dir else None
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = metrics
        self._check_args()

    def _check_args(self):
        if not self.index_file.exists():
            raise FileNotFoundError(f'Index file: {self.index_file} does not exist')
        if not self.edit_video_dir.exists():
            raise FileNotFoundError(f'Edit video directory: {self.edit_video_dir} does not exist')
        if self.reference_video_dir and not self.reference_video_dir.exists():
            raise FileNotFoundError(f'Reference video directory: {self.reference_video_dir} does not exist')
        
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
        output_dir: Optional[Path] = None,
        save_results: bool = True,
        save_summary: bool = True,
        evaluator_kwargs: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Dict[str, List]]:
        # evaluate results for each sample
        results = {}
        for metric in tqdm(self.metrics):
            try:
                module = importlib.import_module(f'.{metric}', package='evaluate')
                evaluator_cls_name = ''.join([w[0].upper() + w[1:] for w in metric.split('_')])
                evaluator_cls = getattr(module, evaluator_cls_name)
                kwargs = evaluator_kwargs[metric] if evaluator_kwargs and metric in evaluator_kwargs else {}

                evaluator = evaluator_cls(
                    index_file=self.index_file,
                    edit_video_dir=self.edit_video_dir,
                    reference_video_dir=self.reference_video_dir,
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
        logger.info(self.reference_video_dir.name)
        logger.info(summary_df)
        
        if output_dir is None and (save_results or save_summary):
            logger.info('Output directory not provided. Results will be saved in the editing video directory')
            output_dir = self.edit_video_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        if save_results:
            merged_results_df.to_csv(output_dir / 'results.csv', index=False)
            logger.info(f'Results saved at: {output_dir / "results.csv"}')
        if save_summary:
            summary_df.to_csv(output_dir / 'summary.csv', index=False)
            logger.info(f'Summary saved at: {output_dir / "summary.csv"}')

        return results