from typing import List, Optional, Sequence, Tuple, Union
import random
import numpy as np

from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
import torch


@TRANSFORMS.register_module()
class GetVisualPromptfromGT(BaseTransform):

    def __init__(self, filter_empty_results: bool = True, max_text_len = 256, max_prompt_num=None,
                 prompt_bboxes_already_transformed: bool = True, mode: str = 'random') -> None:
        self.filter_empty_results = filter_empty_results
        self.max_text_len = max_text_len
        self.max_prompt_num = max_prompt_num if isinstance(max_prompt_num, int) else 1e8
        # NOTE: Whether the bboxes extracted from gt_bboxes have already been
        # transformed by various augmentation pipelines.
        # True for Train, False for Eval and Predict.
        self.prompt_bboxes_already_transformed = prompt_bboxes_already_transformed
        self.mode = mode
        assert self.mode in ['random', 'one-shot', 'full-shot', 'random_and_one_shot']
    
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        if 'prompt_bboxes' in results:                              # prompt_image_test_pipeline for predict
            assert 'prompt_type' in results and results['prompt_type'] == 'Visual'
            if "prompt_bboxes_labels" not in results:
                print("Predict : prompt_bboxes_labels is not specified, set all prompt_bboxes to one category.")
                results['prompt_bboxes_labels'] = [1] * len(results['prompt_bboxes'])
            else:
                assert len(results['prompt_bboxes']) == len(results['prompt_bboxes_labels']), \
                      "The number of prompt_bboxes and prompt_bboxes_labels must be the same"
            # Sort prompt_bboxes by label
            prompt_bboxes_zipped = sorted(zip(results['prompt_bboxes'], results['prompt_bboxes_labels']), key=lambda x: x[1])
            results['prompt_bboxes'] = torch.tensor([x[0] for x in prompt_bboxes_zipped])
            results['prompt_bboxes_labels'] = torch.tensor([x[1] for x in prompt_bboxes_zipped])
            assert results['prompt_bboxes'].shape[0] == results['prompt_bboxes_labels'].shape[0]
            results['prompt_bboxes_already_transformed'] = self.prompt_bboxes_already_transformed
        else:                                                       # train_pipeline / test_pipeline(eval and test)
            assert 'gt_bboxes' in results
            gt_bboxes, gt_bboxes_labels = torch.tensor(results['gt_bboxes'].numpy()), results['gt_bboxes_labels']
            if gt_bboxes.shape[0] > 0:                              # for train and eval
                gt_bboxes_unique_label = np.unique(gt_bboxes_labels)
                prompt_bboxes = []
                prompt_bboxes_label = []
                for gt_bboxes_label in gt_bboxes_unique_label:
                    c_gt_bboxes = gt_bboxes[gt_bboxes_labels==gt_bboxes_label]
                    if self.mode == 'random':       # random num visual prompt, for train or eval
                        c_prompt_bbox_num = random.choice(range(1, min(c_gt_bboxes.shape[0] + 1, self.max_prompt_num)))
                    elif self.mode == 'one-shot':   # one-shot visual prompt, for eval
                        c_prompt_bbox_num = 1
                    elif self.mode == 'full-shot':
                        c_prompt_bbox_num = c_gt_bboxes.shape[0]
                    elif self.mode == 'random_and_one_shot':
                        if random.choice([0, 1]) == 0:      # random num visual prompt
                            c_prompt_bbox_num = random.choice(range(1, min(c_gt_bboxes.shape[0] + 1, self.max_prompt_num)))
                        else:                               # one-shot visual prompt
                            c_prompt_bbox_num = 1
                    else:
                        raise Exception("error type of mode in GetVisualPromptfromGT.")
                    selected_indices = torch.randperm(c_gt_bboxes.shape[0])[:c_prompt_bbox_num]
                    c_prompt_bboxes = torch.index_select(c_gt_bboxes, 0, selected_indices)
                    prompt_bboxes.append(c_prompt_bboxes)
                    prompt_bboxes_label += [gt_bboxes_label] * c_prompt_bboxes.shape[0]
                results['prompt_bboxes'] = torch.cat(prompt_bboxes, dim=0)
                results['prompt_bboxes_labels'] = torch.tensor(prompt_bboxes_label)
                results['prompt_bboxes_already_transformed'] = self.prompt_bboxes_already_transformed
                # NOTE: In visual_prompt_bert, each class adds one prompt bbox.
                # To limit the total prompt bbox count within max_text_len, filter here.
                if results['prompt_bboxes'].shape[0] + len(gt_bboxes_unique_label) >= self.max_text_len:
                    return None
            else:
                if self.filter_empty_results:                       # train for gt_bboxes.shape[0]==0
                    return None
                else:                                               # predict
                    # NOTE: Set fake prompt_bboxes for cases when gt_bboxes is empty
                    results['prompt_bboxes'] = torch.tensor([[10, 10, 100, 100],])
                    results['prompt_bboxes_labels'] = torch.tensor([0])
                    results['prompt_bboxes_already_transformed'] = False
        
        # NOTE: Ensure compatibility with prompt image pipelines
        results['prompt_img_path'] = results['img_path']
        results['prompt_img_shape'] = results['img_shape']
        results['prompt_ori_shape'] = results['ori_shape']

        return results

    def __repr__(self):
        return self.__class__.__name__