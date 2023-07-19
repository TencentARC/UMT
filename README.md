# Unified Multi-modal Transformers

[![DOI](https://badgen.net/badge/DOI/10.1109%2FCVPR52688.2022.00305/blue?cache=300)](https://doi.org/10.1109/CVPR52688.2022.00305)
[![arXiv](https://badgen.net/badge/arXiv/2203.12745/red?cache=300)](https://arxiv.org/abs/2203.12745)
[![License](https://badgen.net/badge/License/BSD%203-Clause%20License?color=cyan&cache=300)](https://github.com/TencentARC/UMT/blob/main/LICENSE)

This repository maintains the official implementation of the paper **UMT: Unified Multi-modal Transformers for Joint Video Moment Retrieval and Highlight Detection** by [Ye Liu](https://yeliu.dev/), Siyuan Li, [Yang Wu](https://scholar.google.com/citations?user=vwOQ-UIAAAAJ), [Chang Wen Chen](https://web.comp.polyu.edu.hk/chencw/), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ), and [Xiaohu Qie](https://scholar.google.com/citations?user=mk-F69UAAAAJ), which has been accepted by [CVPR 2022](https://cvpr2022.thecvf.com/).

<p align="center"><img width="850" src="https://raw.githubusercontent.com/TencentARC/UMT/main/.github/model.svg"></p>

## Installation

Please refer to the following environmental settings that we use. You may install these packages by yourself if you meet any problem during automatic installation.

- CUDA 11.5.0
- CUDNN 8.3.2.44
- Python 3.10.0
- PyTorch 1.11.0
- [NNCore](https://github.com/yeliudev/nncore) 0.3.6

### Install from source

1. Clone the repository from GitHub.

```
git clone https://github.com/TencentARC/UMT.git
cd UMT
```

2. Install dependencies.

```
pip install -r requirements.txt
```

## Getting Started

### Download and prepare the datasets

1. Download and extract the datasets.

- [QVHighlights](https://huggingface.co/yeliudev/UMT/resolve/main/datasets/qvhighlights-a8559488.zip)
- [Charades-STA](https://huggingface.co/yeliudev/UMT/resolve/main/datasets/charades-2c9f7bab.zip)
- [YouTube Highlights](https://huggingface.co/yeliudev/UMT/resolve/main/datasets/youtube-8a12ff08.zip)
- [TVSum](https://huggingface.co/yeliudev/UMT/resolve/main/datasets/tvsum-ec05ad4e.zip)

2. Prepare the files in the following structure.

```
UMT
├── configs
├── datasets
├── models
├── tools
├── data
│   ├── qvhighlights
│   │   ├── *features
│   │   ├── highlight_{train,val,test}_release.jsonl
│   │   └── subs_train.jsonl
│   ├── charades
│   │   ├── *features
│   │   └── charades_sta_{train,test}.txt
│   ├── youtube
│   │   ├── *features
│   │   └── youtube_anno.json
│   └── tvsum
│       ├── *features
│       └── tvsum_anno.json
├── README.md
├── setup.cfg
└── ···
```

### Train a model

Run the following command to train a model using a specified config.

```shell
# Single GPU
python tools/launch.py ${path-to-config}

# Multiple GPUs
torchrun --nproc_per_node=${num-gpus} tools/launch.py ${path-to-config}
```

### Test a model and evaluate results

Run the following command to test a model and evaluate results.

```
python tools/launch.py ${path-to-config} --checkpoint ${path-to-checkpoint} --eval
```

### Pre-train with ASR captions on QVHighlights

Run the following command to pre-train a model using ASR captions on QVHighlights.

```
torchrun --nproc_per_node=4 tools/launch.py configs/qvhighlights/umt_base_pretrain_100e_asr.py
```

## Model Zoo

We provide multiple pre-trained models and training logs here. All the models are trained with a single NVIDIA Tesla V100-FHHL-16GB GPU and are evaluated using the default metrics of the datasets.

<table>
  <tr>
    <th rowspan="2">Dataset</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">Type</th>
    <th colspan="2">MR mAP</th>
    <th colspan="2">HD mAP</th>
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <th>R1@0.5</th>
    <th>R1@0.7</th>
    <th>R5@0.5</th>
    <th>R5@0.7</th>
  </tr>
  <tr>
    <td align="center" rowspan="2">
      <a href="https://arxiv.org/abs/2107.09609">QVHighlights</a>
    </td>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/qvhighlights/umt_base_200e_qvhighlights.py">UMT-B</a>
    </td>
    <td align="center">—</td>
    <td align="center" colspan="2">38.59</td>
    <td align="center" colspan="2">39.85</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_base_200e_qvhighlights-9a13c673.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_base_200e_qvhighlights.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/qvhighlights/umt_base_200e_qvhighlights.py">UMT-B</a>
    </td>
    <td align="center">w/ PT</td>
    <td align="center" colspan="2">39.26</td>
    <td align="center" colspan="2">40.10</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_base_finetune_200e_qvhighlights-d674a657.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_base_finetune_200e_qvhighlights.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center" rowspan="2">
      <a href="https://arxiv.org/abs/1705.02101">Charades-STA</a>
    </td>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/charades/umt_base_va_100e_charades.py">UMT-B</a>
    </td>
    <td align="center">V + A</td>
    <td align="center">48.31</td>
    <td align="center">29.25</td>
    <td align="center">88.79</td>
    <td align="center">56.08</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_base_va_100e_charades-b51a65aa.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_base_va_100e_charades.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/charades/umt_base_vo_100e_charades.py">UMT-B</a>
    </td>
    <td align="center">V + O</td>
    <td align="center">49.35</td>
    <td align="center">26.16</td>
    <td align="center">89.41</td>
    <td align="center">54.95</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_base_vo_100e_charades-39ec9829.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_base_vo_100e_charades.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center" rowspan="6">
      <a href="https://doi.org/10.1007/978-3-319-10590-1_51">YouTube<br>Highlights</a>
    </td>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/youtube/umt_small_100e_youtube_dog.py">UMT-S</a>
    </td>
    <td align="center">Dog</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">65.93</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_dog-90f2189e.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_dog.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/youtube/umt_small_100e_youtube_gym.py">UMT-S</a>
    </td>
    <td align="center">Gymnastics</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">75.20</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_gym-fe749774.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_gym.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/youtube/umt_small_100e_youtube_par.py">UMT-S</a>
    </td>
    <td align="center">Parkour</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">81.64</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_par-4d8a9e8b.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_par.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/youtube/umt_small_100e_youtube_ska.py">UMT-S</a>
    </td>
    <td align="center">Skating</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">71.81</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_ska-f12710a8.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_ska.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/youtube/umt_small_100e_youtube_ski.py">UMT-S</a>
    </td>
    <td align="center">Skiing</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">72.27</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_ski-1ca38d91.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_ski.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/youtube/umt_small_100e_youtube_sur.py">UMT-S</a>
    </td>
    <td align="center">Surfing</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">82.71</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_sur-9be4b575.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_100e_youtube_sur.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center" rowspan="10">
      <a href="https://doi.org/10.1109/cvpr.2015.7299154">TVSum</a>
    </td>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/tvsum/umt_small_500e_tvsum_vt.py">UMT-S</a>
    </td>
    <td align="center">VT</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">87.54</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_vt-3eff6e1b.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_vt.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/tvsum/umt_small_500e_tvsum_vu.py">UMT-S</a>
    </td>
    <td align="center">VU</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">81.51</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_vu-ea40b5ee.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_vu.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/tvsum/umt_small_500e_tvsum_ga.py">UMT-S</a>
    </td>
    <td align="center">GA</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">88.22</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_ga-7217ee96.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_ga.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/tvsum/umt_small_500e_tvsum_ms.py">UMT-S</a>
    </td>
    <td align="center">MS</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">78.81</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_ms-a41636ac.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_ms.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/tvsum/umt_small_500e_tvsum_pk.py">UMT-S</a>
    </td>
    <td align="center">PK</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">81.42</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_pk-4ea24b6c.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_pk.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/tvsum/umt_small_500e_tvsum_pr.py">UMT-S</a>
    </td>
    <td align="center">PR</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">86.96</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_pr-815f527a.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_pr.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/tvsum/umt_small_500e_tvsum_fm.py">UMT-S</a>
    </td>
    <td align="center">FM</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">75.96</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_fm-cf6ebb1d.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_fm.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/tvsum/umt_small_500e_tvsum_bk.py">UMT-S</a>
    </td>
    <td align="center">BK</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">86.89</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_bk-12c75dff.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_bk.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/tvsum/umt_small_500e_tvsum_bt.py">UMT-S</a>
    </td>
    <td align="center">BT</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">84.42</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_bt-3b666738.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_bt.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/TencentARC/UMT/configs/tvsum/umt_small_500e_tvsum_ds.py">UMT-S</a>
    </td>
    <td align="center">DS</td>
    <td align="center" colspan="2">—</td>
    <td align="center" colspan="2">79.63</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_ds-55549243.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_small_500e_tvsum_ds.json">metrics</a>
    </td>
  </tr>
</table>

Here, `w/ PT` means initializing the model using pre-trained [weights](https://huggingface.co/yeliudev/UMT/resolve/main/checkpoints/umt_base_pretrain_100e_asr-ebae4090.pth) on ASR captions. `V`, `A`, and `O` indicate video, audio, and optical flow, respectively.

## Citation

If you find this project useful for your research, please kindly cite our paper.

```bibtex
@inproceedings{liu2022umt,
  title={UMT: Unified Multi-modal Transformers for Joint Video Moment Retrieval and Highlight Detection},
  author={Liu, Ye and Li, Siyuan and Wu, Yang and Chen, Chang Wen and Shan, Ying and Qie, Xiaohu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={3042--3051},
  year={2022}
}
```
