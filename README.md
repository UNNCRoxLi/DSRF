## [**DSRF: A Dynamic and Scalable Reasoning Framework for Solving RPMs (NeurIPS 2025)**](https://openreview.net/forum?id=I3Ep3gQUaw)
This repository contains the official implementation of the paper **"DSRF: A Dynamic and Scalable Reasoning Framework for Solving RPMs"**, accepted by **NeurIPS 2025**. 

**Authors:** Chengtai Li, Yuting He, Jianfeng Ren, Ruibin Bai, Yitian Zhao, Xudong Jiang

**Note:** This is an **Early Access (EA)** version of the code. We are actively working on cleaning up the codebase, improving documentation, and adding more features. Expect frequent updates and potential changes.

## Code environments and toolkits

- OS: Ubuntu 18.04.5
- CUDA: 12.6
- Python: 3.10.18
- Toolkit: PyTorch 2.7.0+cu126
- 2x GPU: NVIDIA RTX A5000
- [thop](https://github.com/Lyken17/pytorch-OpCounter)
- [muon](https://github.com/KellerJordan/Muon)

### Experiments

#### Dataset Structure

Please prepare datasets with the following structure:


```markdown
your_dataset_root_dir/

    ├─I-RAVEN (RAVEN or RAVEN-FAIR)
    │  ├─center_single
    │  ├─distribute_four
    │  ├─distribute_nine
    │  ├─in_center_single_out_center_single
    │  ├─in_distribute_four_out_center_single
    │  ├─left_center_single_right_center_single
    │  └─up_center_single_down_center_single

```

#### Training and Evaluation
You can train different variants of the DSRF model (Small, Medium, Large) and their corresponding lightweight versions by specifying the `--dsrf-scale` argument (`S`, `M`, or `L`) and optionally adding the `--dsrf-light` argument.

> **Note on Optimizer:** We use the [Muon](https://github.com/KellerJordan/Muon) optimizer for hidden layers (parameters with dimensions ≥ 2) and AdamW for others.
> * `--muon-lr`: Learning rate for the Muon optimizer (default: 0.003).
> * `--lr`: Learning rate for the AdamW optimizer (default: 0.001).

**Train DSRF-Small (S) on RAVEN:**

```python
python main.py --dataset-name RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1 --fp16 \
               --image-size 80 --epochs 200 --seed 12345 --batch-size 108 --lr 0.001 --muon-lr 0.003 --wd 1e-5 \
               -a dsrf_raven --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt your_checkpoint_dir \
               --dsrf-scale S\
```

**Train DSRF-Medium (M) on RAVEN:**

```python
python main.py --dataset-name RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1 --fp16 \
               --image-size 80 --epochs 200 --seed 12345 --batch-size 108 --lr 0.001 --muon-lr 0.003 --wd 1e-5 \
               -a dsrf_raven --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt your_checkpoint_dir \
               --dsrf-scale M\
```

**Train DSRF-Small (L) on RAVEN:**

```python
python main.py --dataset-name RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1 --fp16 \
               --image-size 80 --epochs 200 --seed 12345 --batch-size 108 --lr 0.001 --muon-lr 0.003 --wd 1e-5 \
               -a dsrf_raven --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt your_checkpoint_dir \
               --dsrf-scale L\
```

**Train Lightweight Versions of DSRF-Small (L) on RAVEN:**

```python
python main.py --dataset-name RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1 --fp16 \
               --image-size 80 --epochs 200 --seed 12345 --batch-size 108 --lr 0.001 --muon-lr 0.003 --wd 1e-5 \
               -a dsrf_raven --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt your_checkpoint_dir \
               --dsrf-scale L --dsrf-light\
```

**Evaluation:**
To evaluate a trained model on the test set, use the `-e` (or `--evaluate`) flag along with `--resume` to specify the checkpoint path. Use `--show-detail` to print detailed accuracy.
```python
python main.py --dataset-name RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1 --fp16 \
               --image-size 80 -a dsrf_raven \
               -e --resume your_checkpoint_dir/model_best.pth.tar \
               --show-detail
```

## Known Issues and Optimization Strategy

We observed that the model may experience **gradient explosion** on certain hardware configurations. To address this, we implemented the following optimization strategy:

**Partial Randomization (Mid-Training)**: At epoch 90, we employ a strategy to decouple learned features from signal transmission. We **preserve the backbone and core reasoning blocks** but **re-initialize the classifier, reducers, and projections** with extremely low variance. This resets signal amplitude to a safe range, eliminating accumulated instability while allowing the model to fine-tune based on mature semantic representations.

## Special Thanks

We would like to express our gratitude to the following projects and contributors:

* **Framework Base**: This code is built upon the framework provided by [AVR-PredRNet-and-SSPredRNet](https://github.com/ZjjConan/AVR-PredRNet-and-SSPredRNet/tree/main). We thank the authors for their open-source contribution.
* **Code Maintenance**: Special thanks to [hawn999](https://github.com/hawn999) for organizing and maintaining the code.

## Citation

--------

If you find this code useful in your research, please consider citing:
    

    @inproceedings{li2025dsrf,
      title={DSRF: A Dynamic and Scalable Reasoning Framework for Solving RPMs},
      author={Li, Chengtai and He, Yuting and Ren, Jianfeng and Bai, Ruibin and Zhao, Yitian and Jiang, Xudong},
      booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
      year={2025}
    }

