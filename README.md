# PanoSplatt3R

<h2 align="center"><b>PanoSplatt3R: Leveraging Perspective Pretraining for Generalized Unposed Wide-Baseline Panorama Reconstruction</b></h2>

<p align="center">Jiahui Ren, Mochu Xiang, Jiajun Zhu, Yuchao Dai</p>


## TODO

- [x] Evaluation Code
- [x] Training Code
- [ ] Model Checkpoint
- [ ] Test Dataset

## Installation

We conduct all our experiments on NVIDIA L40 48G GPUs.Results may vary slightly across devices or environments, but the overall outcome remains consistent.You can creat a conda environment as following:

```bash
conda create --name panosplatt3r python=3.11.3
conda activate panosplatt3r
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Datasets Preparation

To obtain the HM3D training set, we follow the rendering process described in [Splatter-360](https://github.com/thucz/splatter360/blob/main/preprocess/README.md) to generate panoramic videos, which involve some randomness.The Replica test set can also be obtained from [Splatter-360](https://github.com/thucz/splatter360)

Since the HM3D test set used by Splatter-360 is not publicly available, we will upload our own HM3D test data.

For fixed-baseline evaluation, the corresponding data can be found in [PanSplat](https://github.com/chengzhag/PanSplat/tree/main).

After downloading the datasets, please update the `roots` and `rgb_roots` in `config/dataset`.

## Evaluation

<!-- download weight -->

- Evaluation on HM3D

```bash
bash scripts/evaluate_hm3d.sh
```

- Evaluation on replica
```bash
bash scripts/evaluate_replica.sh
```

- Evaluation on fixed baseline
```bash
bash scripts/evaluate_fixed_baseline.sh
```

- Output visualization can be enabled by revising `to_save_image` in `evaluate/evaluate_random.py` and `evaluate/evaluate_fix_baseline.py`. Rendered results will be saved in the `result/` directory.

- Relative pose evaluation
```bash
scripts/evaluate_pose.sh
```

## Training

Please download the [MAST3R](https://github.com/naver/mast3r) pretrained weights to the `checkpoints` directory, and set `effi_attention` to `true` in `config/model/encoder/backbone/croco.yaml` to enable memory-efficient attention using xFormers.

- Stage 1: Backbone Pretraining

```bash 
bash scripts/run_training_dust3r.sh
```

To load the stage 1 checkpoint, update `weights_path` in `config/defalts.yaml`.

- Stage 2: Full Training

```bash 
bash scripts/run_training_gaussian.sh
```

The training procedure requires at least 48 GB of GPU memory.

## Citation

## Acknowledgements

We acknowledge the valuable support of open-source projects like [NoPoSplat](https://github.com/cvg/NoPoSplat), [Splatter-360](https://github.com/thucz/splatter360), [PanSplat](https://github.com/chengzhag/PanSplat), [PanoGRF](https://github.com/thucz/PanoGRF), [DUSt3R](https://github.com/naver/dust3r), [MAST3R](https://github.com/naver/mast3r).