# Matcha

This is the official implementation of the following paper:

[ICLR 2025] Matcha: Mitigating Graph Structure Shifts with Test-Time Adaptation.
Wenxuan Bao, Zhichen Zeng, Zhining Liu, Hanghang Tong, Jingrui He.

[[ArXiv]](https://arxiv.org/abs/2410.06976)
[[OpenReview]](https://openreview.net/forum?id=EpgoFFUM2q)
[[Poster]](https://github.com/baowenxuan/Matcha/blob/master/material/Matcha_poster.pdf)
[[Slides]](https://github.com/baowenxuan/Matcha/blob/master/material/Matcha_slides.pdf)

**Updates:**

- 06/21/2025: Dataset, core code

More experiment scripts and example results will be available soon. 

## Introduction

Powerful as they are, graph neural networks (GNNs) are known to be vulnerable to distribution shifts. Recently,
test-time adaptation (TTA) has attracted attention due to its ability to adapt a pre-trained model to a target domain,
without re-accessing the source domain. However, existing TTA algorithms are primarily designed for attribute shifts in
vision tasks, where samples are independent. These methods perform poorly on graph data that experience structure
shifts, where node connectivity differs between source and target graphs. We attribute this performance gap to the
distinct impact of node attribute shifts versus graph structure shifts: the latter significantly degrades the quality of
node representations and blurs the boundaries between different node categories. To address structure shifts in graphs,
we propose Matcha, an innovative framework designed for effective and efficient adaptation to structure shifts by
adjusting the htop-aggregation parameters in GNNs. To enhance the representation quality, we design a
prediction-informed clustering loss to encourage the formation of distinct clusters for different node categories.
Additionally, Matcha seamlessly integrates with existing TTA algorithms, allowing it to handle attribute shifts
effectively while improving overall performance under combined structure and attribute shifts. We validate the
effectiveness of Matcha on both synthetic and real-world datasets, demonstrating its robustness across various
combinations of structure and attribute shifts.

## Requirements

```
torch                             2.4.1
torch-geometric                   2.6.1
torch_scatter                     2.1.2
torch_sparse                      0.6.18
```


## Prepare the datasets

Please organize the dataset files according to the following directory structure:

```
Matcha/
├── src/
├── script/
└── data/
    ├── csbm/
    ├── syn-cora/
    ├── syn-products/
    ├── twitch/
    └── ogbn_arxiv/
```

- `Matcha/` is the project root directory.
- `data/` contains all datasets used in the project.
- `csbm/`, `syn-cora`, etc. are subdirectories for specific datasets.

### CSBM Datasets

We adapt the code from the [GPRGNN github repo](https://github.com/jianhao2016/GPRGNN) to generate the CSBM datasets.
You can also generate the data by running

```shell
cd src
python csbm_gen.py
```

Notice that it takes a while to generate the dataset due to the low-efficiency for loop.

We also provide a copy [here](https://drive.google.com/drive/folders/1h5411e3oPv1MDnPwb7SkPIl4wFF6maCV?usp=drive_link).

### Syn-Cora & Syn-Products

We originally downloaded these two datasets in `npz` format from the
[H2GCN github repo](https://github.com/GemsLab/H2GCN/tree/master/npz-datasets).
However, we recently find that we do not have the access to their Google Drive anymore.
If you have the same issue, you may download the data from our copy
[here](https://drive.google.com/drive/folders/1h5411e3oPv1MDnPwb7SkPIl4wFF6maCV?usp=drive_link).
Please download the `tar.gz` file to your `${DATA}` path, and extract them.

For Syn-Cora, we observe that different homophily levels and seed settings share identical node features (up to index
shuffling). This introduces data leakage problem: Models like MLP can overfit the node features and achieve high
performance without using edge information. To prevent such data leakage, we adopt a non-overlapping train-test node
split: For each class, we use 25% as the training nodes and 75% as the testing nodes.

For Syn-Products, node features are sampled from the much larger ogbn-products graph. As a result, we did not observe
significant feature overlap across graphs with different homophily levels and seeds. Therefore, we do not apply any
masking on Syn-Products.

### Twitch & OGB-Arxiv

We use the implementation from the [EERM github repo](https://github.com/qitianwu/GraphOOD-EERM).

## Algorithm

The core code of Matcha is provided in `src/algo/Matcha.py`. 

## Experiments

Example script for CSBM dataset is provided in `script/csbm/`. Please run experiments with the following steps: 

1. Run `pretrain.sh` to get pretrained weights for each setting. 
2. Run `homo2hetero.sh` to test Matcha or its combination with base TTA methods on each setting. 

## Citation

```
@inproceedings{bao2025matcha,
  author       = {Wenxuan Bao and
                  Zhichen Zeng and
                  Zhining Liu and
                  Hanghang Tong and
                  Jingrui He},
  title        = {Matcha: Mitigating Graph Structure Shifts with Test-Time Adaptation},
  booktitle    = {The Thirteenth International Conference on Learning Representations,
                  {ICLR} 2025, Singapore, April 24-28, 2025},
  publisher    = {OpenReview.net},
  year         = {2025},
  url          = {https://openreview.net/forum?id=EpgoFFUM2q},
}
```



