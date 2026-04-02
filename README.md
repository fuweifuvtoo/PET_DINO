# PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training
This is the official implementation of [PET-DINO](https://arxiv.org/abs/2604.00503).

🎉🎉🎉 Our paper is accepted by CVPR 2026, congratulations and many thanks to the co-authors!

If you find our work helpful, please kindly give us a star⭐

## UPDATES

* [2026.04.05] Release the code.
* [2026.04.03] The PET-DINO paper is now publicly available on arXiv.
* [2026.02.21] 🔥🔥🔥 PET-DINO is accepted by CVPR 2026! Congratulations and many thanks to the co-authors!

## Introduction

PET-DINO is an open-vocabulary object detector that supports **text prompts** and **visual prompts**. It is built upon [MM-Grounding-DINO](https://arxiv.org/abs/2401.02361), incorporating visual prompt capabilities for flexible and expressive object detection.

PET-DINO supports two types of prompts:

- **Text Prompt**: Detect objects described by text labels (e.g., "zebra . giraffe . bird").
- **Visual Prompt**: Detect objects using visual information as prompts, which can be evaluated in two modes:
  - **Visual-I**
  - **Visual-G**

## Environment

Please first install following the instructions in the [get_started](https://github.com/open-mmlab/mmdetection/blob/main/docs/en/get_started.md) section, then you need to install additional dependency packages:

```bash
pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

> **Note**: The LVIS third-party library does not currently support numpy >= 1.24. Please ensure your numpy version meets the requirements. It is recommended to install `numpy==1.23`.

## Data Preparation

### Pretrained Weights

Download the following pretrained models and place them under `pretrained/`:

| Model | Path |
| --- | --- |
| Swin-T | `pretrained/swin/swin_tiny_patch4_window7_224.pth` |
| Swin-L | `pretrained/swin/swin_large_patch4_window7_224.pth` |
| BERT | `pretrained/bert-base-uncased/` |
| [MM-Grounding-DINO Swin-T](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth) | `pretrained/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth` |
| [MM-Grounding-DINO Swin-L](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth) | `pretrained/grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth` |

For downloading BERT weights, please refer to the [MM-Grounding-DINO usage guide](configs/mm_grounding_dino/usage.md#download-bert-weight).

### Datasets

For detailed data preparation instructions, please refer to the [MM-Grounding-DINO dataset preparation guide](configs/mm_grounding_dino/dataset_prepare.md).

**Training data**: [Objects365v1](configs/mm_grounding_dino/dataset_prepare.md#1-objects365v1)

**Evaluation datasets**:
- [COCO 2017](configs/mm_grounding_dino/dataset_prepare.md#1-coco-2017)
- [LVIS 1.0 (minival)](configs/mm_grounding_dino/dataset_prepare.md#2-lvis-10)
- [ODinW35](configs/mm_grounding_dino/dataset_prepare.md#3-odinw)

The overall data directory structure should look like this:

```text
data/
├── objects365v1/
│   ├── objects365_train.json
│   ├── objects365_val.json
│   ├── o365v1_train_od.json
│   ├── o365v1_label_map.json
│   ├── train/
│   └── val/
├── coco/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   ├── train2017/
│   └── val2017/
├── lvis/
│   ├── annotations/
│   │   ├── lvis_v1_train.json
│   │   ├── lvis_v1_val.json
│   │   ├── lvis_v1_minival_inserted_image_name.json
│   ├── train2017/
│   └── val2017/
└── odinw/
    ├── AerialMaritimeDrone/
    │   ├── large/
    │   │   ├── test/
    │   │   ├── train/
    │   │   └── valid/
    │   └── tiled/
    ├── AmericanSignLanguageLetters/
    ├── Aquarium/
    └── ...  (35 datasets in total)
```

## Train

```bash
bash tools/dist_train.sh configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py 8 --auto-scale-lr
```

## Evaluation

### COCO

```bash
# Visual-I
CONFIG=configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Visual'

# Text
CONFIG=configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Text'

# Visual-G (extract embedding first, then evaluate)
python scripts/image_demo.py data/coco/train2017 \
    $CONFIG --weights $CHECKPOINT \
    --prompt_type 'Visual' \
    --input_od_json 'images/cocoTrain_odvg_16.json' \
    --extract-visual-embedding --out-dir ./outputs/visual_embedding_coco/

CONFIG=configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Visual' \
    test_cfg.type='CustomTestLoop' \
    test_cfg.prompt_visual_embedding_path='./outputs/visual_embedding_coco/'
```

### LVIS minival

```bash
# Visual-I
CONFIG=configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_lvis_minival_vi.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Visual'

# Text
CONFIG=configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_lvis_minival_vg_text.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Text'

# Visual-G
python scripts/image_demo.py data/lvis \
    $CONFIG --weights $CHECKPOINT \
    --prompt_type 'Visual' \
    --input_od_json 'images/lvisTrain_odvg_16.json' \
    --extract-visual-embedding --out-dir ./outputs/visual_embedding_lvis/

CONFIG=configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_lvis_minival_vg_text.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Visual' \
    test_cfg.type='CustomTestLoop' \
    test_cfg.prompt_visual_embedding_path='./outputs/visual_embedding_lvis/'
```

### ODinW35

```bash
# Visual-I
CONFIG=configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_odinw35.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Visual'
# Manually copy the output to the `sample_text` variable in `scripts/extract_bbox_map_from_odinw_str_results.py`, then run:
python scripts/extract_bbox_map_from_odinw_str_results.py

# Text
CONFIG=configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_odinw35.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Text'
# Manually copy the output to the `sample_text` variable in `scripts/extract_bbox_map_from_odinw_str_results.py`, then run:
python scripts/extract_bbox_map_from_odinw_str_results.py

# Visual-G
CONFIG=./configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py
ODinW35_CONFIG=./configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_odinw35.py
bash scripts/evaluate_visual-G_of_odinw35.sh $CONFIG $ODinW35_CONFIG $CHECKPOINT 8
# Calculate the average mAP across all ODINW datasets
python scripts/get_avg_map_of_odinw.py
```

## Inference

### Text Prompt

```bash
python scripts/image_demo.py images/animals.png \
    configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py \
    --weights $CHECKPOINT --pred-score-thr 0.3 \
    --texts 'zebra. giraffe. bird' -c
```

### Visual Prompt (Bounding Boxes)

```bash
python scripts/image_demo.py images/animals.png \
    configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py \
    --weights $CHECKPOINT --pred-score-thr 0.3 \
    --prompt_type 'Visual' \
    --prompt_bboxes '[[1291.6, 679.9, 1536.8, 840.0], [845.0, 181.9, 1210.2, 723.5], [638.0, 470.9, 815.0, 704.4]]' \
    --prompt_bboxes_labels '[30, 2, 46]'
```

### Visual Prompt (Visual Embeddings)

```bash
# Extract visual embedding (single image)
python scripts/image_demo.py images/animals.png \
    configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py \
    --weights $CHECKPOINT \
    --prompt_type 'Visual' \
    --prompt_bboxes '[[1291.6, 679.9, 1536.8, 840.0], [845.0, 181.9, 1210.2, 723.5], [638.0, 470.9, 815.0, 704.4]]' \
    --prompt_bboxes_labels '[30, 2, 46]' \
    --extract-visual-embedding --out-dir ./outputs/visual_embedding_animals/

# Inference with visual embedding
python scripts/image_demo.py images/animals.png \
    configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py \
    --weights $CHECKPOINT --pred-score-thr 0.3 \
    --prompt_type 'Visual' \
    --prompt_visual_embedding_path outputs/visual_embedding_animals/30.pt
```

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@misc{fu2026petdinounifyingvisualcues,
      title={PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training}, 
      author={Weifu Fu and Jinyang Li and Bin-Bin Gao and Jialin Li and Yuhuan Lin and Hanqiu Deng and Wenbing Tao and Yong Liu and Chengjie Wang},
      year={2026},
      eprint={2604.00503},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.00503}, 
}
```

## Acknowledgement

This project is built upon the following excellent works:

- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MM-Grounding-DINO](https://arxiv.org/abs/2401.02361): An open and comprehensive pipeline for unified object grounding and detection.

## License

This project is released under the [Apache 2.0 license](LICENSE).
