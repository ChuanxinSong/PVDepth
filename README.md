<div align="center">
<h1>PVDepth: Panoramic Video Depth Estimation via Geometry-Aware Spatiotemporal Adaptation</h1>

<img src="https://img.shields.io/badge/ICML-2026-4b44ce.svg" alt="ICML 2026">
<a href="https://openreview.net/pdf?id=HwYgnZjxUB"><img src="https://img.shields.io/badge/Paper-OpenReview-b31b1b.svg" alt="Paper"></a>
<a href="https://huggingface.co/datasets/Soon122/PanoCARLA"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PanoCARLA_Dataset-blue" alt="Dataset"></a>
<a href="https://huggingface.co/Soon122/PVDepth"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_Weights-orange' alt="Weights"></a>

<a href="https://scholar.google.com/citations?user=NhZxUX0AAAAJ&hl=zh-CN">Chuanxin Song</a>,
<a href="https://scholar.google.com/citations?user=CFMuFGoAAAAJ&hl=zh-CN">Peixi Peng</a>
</div>

🤗 If you find **PVDepth** or **PanoCARLA** useful, **please help ⭐ this repo**, which is important to open-source projects. Thanks!

## Introduction

We introduce **[PanoCARLA](https://huggingface.co/datasets/Soon122/PanoCARLA)**, a large-scale synthetic RGB-D panoramic video dataset, and **PVDepth**, a generative framework for panoramic video depth estimation. By addressing ERP-specific spatial distortions and temporal non-linear dynamics, PVDepth produces accurate and temporally consistent depth sequences.

## Poster

<p align="center">
  <img src="asset/PVDepth_Poster.png" alt="PVDepth ICML 2026 Poster" width="100%">
</p>

## Checklist / TODOs

- [x] Release the PanoCARLA dataset
- [x] Release the training code and evaluation scripts
- [x] Release the model weights
- [ ] Open-source the panoramic data collection pipeline

## Installation

```bash
git clone https://github.com/ChuanxinSong/PVDepth.git
cd PVDepth

conda create -n pvdepth python=3.10 -y
conda activate pvdepth

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Quick Start

`run_infer_any.sh` supports a single image, an image folder, or a video as input. For example:

```bash
GPU_ID=0 bash run_infer_any.sh examples/200035.mp4
```

The output directory contains the relative inverse-depth prediction as a NumPy
file and a PNG or MP4 visualization.

Processing a 110-frame video at 1024 × 512 resolution requires approximately
31 GB of GPU memory. If GPU memory is insufficient, set `CPU_OFFLOAD` in
`run_infer_any.sh` to `model`, or use `sequential` for a further substantial
reduction to approximately 10 GB of GPU memory. Sequential CPU offloading is
considerably slower.

### Model Checkpoint

The default checkpoint, [Soon122/PVDepth](https://huggingface.co/Soon122/PVDepth),
is downloaded automatically on first use. To use a manually downloaded or
custom checkpoint, pass its UNet directory as the second argument:

```bash
GPU_ID=0 bash run_infer_any.sh <input_path> path/to/unet [output_dir]
```

## Dataset

The [PanoCARLA dataset](https://huggingface.co/datasets/Soon122/PanoCARLA) is available on Hugging Face.

Download the dataset with the Hugging Face CLI:

```bash
hf auth login
hf download Soon122/PanoCARLA \
  --repo-type dataset \
  --local-dir path/to/PanoCARLA

tar -xf path/to/PanoCARLA/town0210_res1024_512.tar \
  -C path/to/PanoCARLA
```

For training, set `data_root` to `path/to/PanoCARLA` and `h5_data_root` to
`path/to/PanoCARLA/panocarla_h5`.

```bash
cp paths.example.env paths.env
```

Update the paths in `paths.env` before starting training.

## Training

PVDepth uses a two-stage training procedure:

1. Run Stage 1 to adapt the model to the sphere-aware noise input introduced
   by PSNI:

   ```bash
   bash train_stage1_psni.sh
   ```

2. Set `UNET_PATH` in `train_stage2_pvdepth.sh` to the Stage 1 UNet checkpoint.

3. Run Stage 2:

   ```bash
   bash train_stage2_pvdepth.sh
   ```

## Benchmark Evaluation

First, run inference on the Town02/Town10 benchmark with the default PVDepth
checkpoint:

```bash
bash run_infer_town0210.sh
```

Then evaluate the generated predictions:

```bash
bash depth_eval/run_eval.sh
```

Evaluation CSV files are saved to `depth_eval/results/pvdepth` by default.
Common settings can be overridden with environment variables:

```bash
GPU_ID=1 \
PRED_BASE_DIR=path/to/predictions \
OUTPUT_DIR=path/to/evaluation_results \
bash depth_eval/run_eval.sh
```

To evaluate a custom checkpoint, set `UNET_PATH` and `OUTPUT_ROOT_DIR` in
`run_infer_town0210.sh`, run inference, and pass the matching output directory
through `PRED_BASE_DIR`.

## Visualization Comparison

The center panel shows the input panoramic video, while the surrounding panels
compare depth predictions from PVDepth (top left), ViPE (top right), DA-2
(bottom left), and UniK3D (bottom right). PVDepth produces stable and
temporally consistent depth predictions throughout the video.

https://github.com/user-attachments/assets/4edbc5f8-fd76-42f8-bcf8-7ed343957ad8

## Limitations

- **Specular reflections.** PVDepth struggles with strongly reflective
  surfaces, such as the water scene at approximately 00:42 in the comparison
  video. Existing depth models also produce invalid or implausible predictions
  in this case, indicating that strong specular reflections remain a shared
  challenge for current depth estimation methods.
- **Indoor evaluation.** PanoCARLA contains only outdoor scenes. Although
  PVDepth produces visually reasonable results on the indoor examples shown in
  the video, we have not conducted quantitative evaluation on indoor datasets.

## Acknowledgements

Our implementation builds upon excellent open-source projects, including but not limited to [DepthCrafter](https://github.com/Tencent/DepthCrafter), [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend), and [Diffusers](https://github.com/huggingface/diffusers).

## Citation

If you find this repository helpful, please consider citing:

```bibtex
@inproceedings{song2026pvdepth,
  author    = {Song, Chuanxin and Peng, Peixi},
  title     = {PVDepth: Panoramic Video Depth Estimation via Geometry-Aware Spatiotemporal Adaptation},
  booktitle = {ICML},
  year      = {2026}
}
```

## License

PVDepth is released primarily for academic research. Components derived from DepthCrafter remain subject to the original DepthCrafter license.

See [LICENSE](LICENSE) and [LICENSE-PVDEPTH](LICENSE-PVDEPTH) for details.
