# OneDiffusion

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange)]()
[![Homepage](https://img.shields.io/badge/Homepage-Offline-gray)]()
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/pdf/2411.16318)

<p align="center">
  <img src="assets/teaser.png" alt="Teaser Image" width="800">
</p>

## Introduction

This is official repo of OneDiffusion, a versatile, large-scale diffusion model that seamlessly supports bidirectional image synthesis and understanding across diverse tasks. We will release the code and checkpoints in early December.

## Qualitative Results

### 1. Text-to-Image
<p align="center">
  <img src="assets/text2image.jpg" alt="Text-to-Image results" width="800">
</p>


### 2. ID customization

<p align="center">
  <img src="assets/onediffusion_appendix_faceid.jpg" alt="ID customization" width="800">
</p>

<p align="center">
  <img src="assets/onediffusion_appendix_faceid_3.jpg" alt="ID customization non-human subject" width="800">
</p>

### 3. Multiview generation

Single image to multiview:

<p align="center">
  <img src="assets/onediffusion_appendix_multiview.jpg" alt="Image to multiview" width="800">
</p>

<p align="center">
  <img src="assets/onediffusion_appendix_multiview_2.jpg" alt="image to multiview" width="800">
</p>

Text to multiview:

<p align="center">
  <img src="assets/text2multiview.jpg" alt="Text to multiview image" width="800">
</p>

### 4. Condition-to-Image and vice versa
<p align="center">
  <img src="assets/cond_and_image.jpg" alt="Condition and Image" width="800">
</p>

### 5. Subject-driven generation

We finetuned the model on [[Subject-200K]](https://huggingface.co/datasets/Yuanshi/Subjects200K) dataset.

<p align="center">
  <img src="assets/subject_driven.jpg" alt="Subject driven generation" width="800">
</p>

## Citation

```bibtex
@misc{le2024diffusiongenerate,
      title={One Diffusion to Generate Them All}, 
      author={Duong H. Le and Tuan Pham and Sangho Lee and Christopher Clark and Aniruddha Kembhavi and Stephan Mandt and Ranjay Krishna and Jiasen Lu},
      year={2024},
      eprint={2411.16318},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.16318}, 
}
```

## Acknowledgements
