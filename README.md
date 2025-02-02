# Generative Deep Learning (2nd Edition) - PyTorch Implementation

This repository contains a PyTorch implementation of the companion code from *Generative Deep Learning (2nd Edition)* by David Foster. The original code, written in Keras, is available [here](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main).

## Overview

The book *Generative Deep Learning (2nd Edition)* explores various generative models, including GANs, VAEs, autoregressive models, and diffusion models. This repository provides equivalent PyTorch implementations of the original Keras-based models.

## Currently Implemented Models
(More models to be added soon)
- **Chapter 2:**
  - MLP
  - CNN
- **Chapter 3:**
  - Autoencoder
  - VAE (variational auto encoder)
- **Chapter 4:**
  - GAN (Generative Adversarial Networks)
  - WGAN-GP (Wasserstein GAN with Gradient Penality)
  - CGAN (conditional GAN)
- **Chapter 5:**
  - LSTM (Long Short-Term memory network)
- **Chapter 8:**
  - DDM (Denoising Diffusion Models)
 
  

## Repository Structure

```
├── data/                 # Dataset storage
│   ├── .gitkeep         
│
├── notebooks/            # Jupyter notebooks for model training and visualization
│   ├── 02_deeplearning/
│   │   ├── 01_mlp/
│   │   │   ├── mlp_pytorch.ipynb
│   │   ├── 02_cnn/
│   │   │   ├── cnn_pytorch.ipynb
│   ├── 03_vae/
│   │   ├── 01_autoencoder/
│   │   │   ├── autoencoder_pytorch.ipynb
│   │   ├── 02_vae_fashion/
│   │   │   ├── vae_fashion_pytorch.ipynb
│   │   ├── 03_vae_faces/
│   │   │   ├── vae_faces_pytorch.ipynb
│   │   │   ├── vae_utils.py
│   ├── 04_gan/
│   │   ├── 01_dcgan/
│   │   │   ├── dcgan_pytorch.ipynb
│   │   ├── 02_wgan_gp/
│   │   │   ├── wgan_gp_pytorch.ipynb
│   │   ├── 03_cgan/
│   │   │   ├── cgan_pytorch.ipynb
│   ├── 05_autoregressive/
│   │   ├── 01_lstm/
│   │   │   ├── lstm_pytorch.ipynb
│   ├── 08_diffusion/
│   │   ├── 01_ddm/
│   │   │   ├── DDM_pytorch.ipynb
│   ├── utils.py # Utility file copied from orginal repo and modified for Pytorch
│
├── scripts/              # Utility scripts copied from the orginal repo
│   ├── downloaders/
│   │   ├── download_bach_cello_data.sh
│   │   ├── download_bach_chorale_data.sh
│   │   ├── download_kaggle_data.sh
│   │   ├── download.sh
│   │   ├── format.sh
│   │   ├── tensorboard.sh
│   │   ├── .gitignore
│
└── README.md             # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Generative_Deep_Learning_PyTorch.git
   cd Generative_Deep_Learning_PyTorch
   ```

(Details on creating Conda enviroment and requirements file comming soon)

## Differences from the Original Keras Implementation
- The code is fully implemented in PyTorch rather than TensorFlow/Keras.
- Model architectures remain as close as possible to the original versions.
- Training loops and utilities have been rewritten to align with PyTorch best practices.
- Some optimizations and improvements specific to PyTorch have been included.

## References
- **Book**: *Generative Deep Learning (2nd Edition)* by David Foster
- **Original Code Repository**: [Keras Implementation](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main)

## Contributing
Contributions are welcome! If you find any issues or would like to improve the implementations, feel free to submit a pull request.

## Contact
mary.raymond.n@gmail.com

If you have any questions or suggestions, feel free to open an issue or reach out!

