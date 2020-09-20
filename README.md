# A Paddle Implementation of StyleGAN (Unofficial)

![Github](https://img.shields.io/badge/Paddle-v1.8.3-green.svg?style=for-the-badge&logo=data:image/png)
![Github](https://img.shields.io/badge/python-3.7-green.svg?style=for-the-badge&logo=python)
![Github](https://img.shields.io/badge/status-AlmostFinished-blue.svg?style=for-the-badge&logo=fire)
![Github](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge&logo=fire)

This repository contains a Paddle implementation of the following paper:
> **A Style-Based Generator Architecture for Generative Adversarial Networks**<br>
> Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA)<br>
> http://stylegan.xyz/paper
>
> **Abstract:** *We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture. Finally, we introduce a new, highly varied and high-quality dataset of human faces.*


![Teaser image](utils/stylegan-teaser.png)
Picture: These people are not real – they were produced by our generator that allows control over different aspects of the image.

## Motivation
There is no styleGAN paddle implementation to many better knowledge. I notice there are great advance in GAN domain and it could be a powerful tool for Auto defect recognition(ADR). I joined a team working on inspection image sponsored by Baidu AI, and hope to come up with a general solution.

## System Requirements
- Windows 10
- Paddle1.8.3-GPU
- opencv-python
- tqdm
- GTX 1080Ti or above

## Training

``` python
# ① pass your own dataset of training, batchsize and common settings in TrainOpts of `opts.py`.

# ② run train_stylegan.py
python3 train_stylegan.py

# ③ you can get intermediate pics generated by stylegenerator in `opts.det/images/`
```

## Finished
- baseline 2020/9/20
- 


## ToDO
- align with Tensorflow implementation
- Multi-GPU implementation
- Results record

## Related
[1. StyleGAN - Official TensorFlow Implementation](https://github.com/NVlabs/stylegan)

[2. StyleGAN_PyTorch](https://github.com/tomguluson92/StyleGAN_PyTorch)
