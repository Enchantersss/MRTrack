# MRTrack: Multi-Expert Motion Reasoning for Identity-Consistent Multi-Object Tracking

This repository contains the beta code release for **MRTrack**, an online query-based multi-object tracker designed for dense scenes with similar appearances, short occlusions, and non-linear or coordinated motion.

MRTrack builds on the MOTR/MOTRv2 and Deformable DETR line of work. It keeps the simple end-to-end tracking pipeline while adding two components:

- **Motion Aware Experts (MAE)**: early decoder layers decode multiple inertia-guided reference-point hypotheses in parallel and fuse them with a query-adaptive gate.
- **Identity latching**: training-time matching uses an annealed inheritance prior and an identity-margin regularizer to discourage unstable frame-wise identity reassignment.

Current experiments report **67.3 HOTA / 68.8 IDF1 / 55.5 AssA** on DanceTrack test under the no-extra-training-data protocol, and **70.0 HOTA** on SportsMOT without extra training data.

## Method Overview

MRTrack follows an online tracking-by-query formulation. For each frame, the model combines:

- track queries propagated from the previous frame,
- detection queries initialized from YOLOX proposals,
- learned slot queries for newly appearing or missed objects.

The query set is decoded by a mixed Deformable Transformer decoder. The default configuration uses **3 MAE layers + 3 standard deformable decoder layers**.

### Motion Aware Experts

For each active track query, MRTrack maintains a momentum-smoothed velocity in normalized image coordinates:

```text
v_t = beta * v_{t-1} + (1 - beta) * delta_t
```

Only the box center is extrapolated; width and height are kept from the previous observation to avoid scale noise. Each MAE layer uses 4 motion experts with reference points ranging from near-static to constant-velocity estimates:

```text
RP
RP + 1/3 * velocity * dt
RP + 2/3 * velocity * dt
RP + velocity * dt
```

Each branch performs deformable cross-attention around its own reference point. A softmax gate then fuses the expert outputs per query, allowing the decoder to keep several plausible motion hypotheses instead of committing to a single stale reference point.

### Identity Latching

During training, MRTrack biases assignment toward temporal identity consistency. Previously matched query-object pairs are probabilistically latched early in training and gradually released as matching becomes more reliable.

The default experiment setting uses:

- latch probability: `0.9 -> 0.3`,
- persistence bias: `2.0 -> 0.5`,
- identity margin: `0.2`,
- identity regularizer weight: `0.2`.

At inference time there is no explicit Hungarian matching or latching. The learned identity prior, propagated track queries, and simple score-based track management maintain online identities.

## Main Results

### DanceTrack Test

Results below follow the no-extra-training-data protocol.

| Method | HOTA | MOTA | IDF1 | AssA | DetA |
| --- | ---: | ---: | ---: | ---: | ---: |
| MOTR | 54.2 | 79.7 | 51.5 | 40.2 | 73.5 |
| ByteTrack | 47.3 | 89.5 | 52.5 | 31.4 | 71.6 |
| Hybrid-SORT | 65.7 | 91.8 | 67.4 | 52.6 | 82.2 |
| MOTRv2 without CrowdHuman | 65.2 | 87.6 | 64.3 | 52.1 | 81.8 |
| **MRTrack** | **67.3** | **90.7** | **68.8** | **55.5** | **81.9** |

### DanceTrack Validation Ablation

| Configuration | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 59.6 | 74.4 | 47.9 | 81.9 | 59.9 | 1969 | 4590 |
| + ID Latching | 61.7 | 75.9 | 50.4 | 84.4 | 62.7 | 1583 | 4057 |
| + MAE Decoder | 62.5 | 75.8 | 51.7 | 83.9 | 63.2 | 1772 | 3913 |
| **+ MAE + ID Latching** | **63.5** | **76.1** | **53.1** | **84.8** | **64.8** | **1550** | **3758** |

### SportsMOT Test

| Method | HOTA | MOTA | IDF1 | AssA | DetA |
| --- | ---: | ---: | ---: | ---: | ---: |
| FairMOT | 49.3 | 86.4 | 53.5 | 34.7 | 70.2 |
| QDTrack | 60.4 | 90.1 | 62.3 | 47.2 | 77.5 |
| ByteTrack | 62.8 | 94.1 | 69.8 | 51.2 | 77.1 |
| SambaMOTR | 69.8 | 90.3 | 71.9 | 59.4 | 82.2 |
| **MRTrack** | **70.0** | **91.8** | **70.9** | **58.7** | **83.6** |

## Installation

Create an environment and install PyTorch according to your CUDA version. The original experiments used PyTorch 2.1.0, while this beta code still follows the MOTRv2/Deformable DETR extension layout.

```bash
conda create -n mrtrack python=3.8
conda activate mrtrack
pip install -r requirements.txt
```

Build the MultiScaleDeformableAttention operator:

```bash
cd models/ops
sh make.sh
cd ../..
```

## Data Preparation

Prepare DanceTrack and CrowdHuman-style detector proposals under one root directory. 本文并没有使用CrowdHuman参与训练，复现时可以忽视这一数据集。 The code expects `--mot_path` to point to this root. The tree below is illustrative; adjust it to your machine.

```text
/data/Dataset/mot/
|-- DanceTrack/
|   |-- train/
|   |-- val/
|   `-- test/
|-- crowdhuman/
|   |-- annotation_train.odgt
|   |-- annotation_val.odgt
|   |-- annotation_trainval.odgt
|   `-- Images/
`-- det_db_motrv2.json
```

If needed, create the CrowdHuman trainval annotation with:

```bash
cat annotation_train.odgt annotation_val.odgt > annotation_trainval.odgt
```

The `det_db_motrv2.json` file stores detector proposals used to initialize detection queries. It should match the paths expected by `datasets/dance.py` and `submit_dance.py`.

## Training

Edit `configs/motrv2.args` before training:

- set `--mot_path` if your dataset root is not the parser default `/root/autodl-tmp/data/Dataset/mot`;
- set `--pretrained` to a Deformable DETR or compatible MOTR/MOTRv2 initialization checkpoint;
- the provided config follows the default experiment setting with 50 epochs, a learning-rate drop at epoch 40, and `--ids-margin-gamma 0.2`.

For quick smoke tests, reduce `--epochs` and `--lr_drop` locally.

Train with 8 GPUs:

```bash
./tools/train.sh configs/motrv2.args
```

The script writes runs under `exps/motrv2/run*/`.

## Inference

Run DanceTrack inference with a trained checkpoint:

```bash
./tools/simple_inference.sh path/to/checkpoint.pth
```

Or evaluate an experiment directory:

```bash
./tools/eval_dance.sh exps/motrv2/run1
```

Predictions are written to a `tracker/` directory and can be submitted or evaluated with the DanceTrack/TrackEval protocol.

## Useful Configuration Flags

```text
--mi-enable                         enable Motion Aware Experts
--mi-num-branches 4                 use four motion expert branches
--mi-layout 3+3                     use 3 MAE layers followed by 3 standard decoder layers
--mi-reference-mode extrapolated    use inertia-extrapolated branch references
--mi-fusion softmax                 use query-adaptive softmax expert fusion
--gm-enable                         enable guided identity latching
--lock-p-init 0.9
--lock-p-final 0.3
--gm-alpha-init 2.0
--gm-alpha-final 0.5
--ids-margin-enable
--ids-margin-m 0.2
--ids-margin-gamma 0.2
```

## Notes

- This is a beta cleanup of an older research codebase.
- Checkpoints and detector proposal files are not included.
- Paths in the provided config may still reflect local experiment machines and should be edited before use.
- The code preserves the MOTRv2-style detector-proposal pipeline and Deformable DETR CUDA extension.

## Acknowledgements

This codebase is built on and inspired by:

- [MOTRv2](https://github.com/megvii-research/MOTRv2)
- [MOTR](https://github.com/megvii-research/MOTR)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [DanceTrack](https://github.com/DanceTrack/DanceTrack)
