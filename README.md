# Fourier-RWKV: A Multi-State Perception Network for Efficient Image Dehazing
Lirong Zheng, Yanshan Li, Rui Yu, Kaihao Zhang

## Paper Link
ArXiv: https://arxiv.org/abs/2512.08161

## Abstract
Image dehazing is crucial for reliable visual perception, yet it remains highly challenging under real-world non-uniform haze conditions. Although Transformer-based methods excel at capturing global context, their quadratic computational complexity hinders real-time deployment. To address this, we propose Fourier Receptance Weighted Key Value (Fourier-RWKV), a novel dehazing framework based on a Multi-State Perception paradigm. The model achieves comprehensive haze degradation modeling with linear complexity by synergistically integrating three distinct perceptual states:
(1) Spatial-form Perception, realized through the Deformable Quad-directional Token Shift (DQ-Shift) operation, which dynamically adjusts receptive fields to accommodate local haze variations; (2) Frequency-domain Perception, implemented within the Fourier Mix block, which extends the core WKV attention mechanism of RWKV from the spatial domain to the Fourier domain, preserving the long-range dependencies essential for global haze estimation while mitigating spatial attenuation; (3) Semantic-relation Perception, facilitated by the Semantic Bridge Module (SBM), which utilizes Dynamic Semantic Kernel Fusion (DSK-Fusion) to precisely align encoder-decoder features and suppress artifacts.
Extensive experiments on multiple benchmarks demonstrate that Fourier-RWKV delivers state-of-the-art performance across diverse haze scenarios while significantly reducing computational overhead, establishing a favorable trade-off between restoration quality and practical efficiency.

## Overview
![](https://github.com/Dilizlr/Fourier-RWKV/blob/main/README_images/overview.png)
