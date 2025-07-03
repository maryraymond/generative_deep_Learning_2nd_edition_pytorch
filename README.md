
# 🎨 (Unofficial) PyTorch Implementation of Generative Deep Learning (2nd Edition)

This repository contains an **unofficial PyTorch implementation** of the companion code from *Generative Deep Learning (2nd Edition)* by David Foster. The original code, written in Keras, is available [here](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main).

The implementation was intentionally as close as possible to the original Keras code in terms of structure, naming conventions, and flow to support side-by-side comparison and study.

---

## 📚 Table of Contents
- [Overview](#overview)
- [Currently Implemented Models](#currently-implemented-models)
- [Repository Structure](#repository-structure)
- [Installation and Usage](#installation-and-usage)
- [Differences from the Original Keras Implementation](#differences-from-the-original-keras-implementation)
- [License](#license)
- [References](#references)
- [Contributing](#contributing)
- [Contact](#contact)

---

## 📖 Overview

The book *Generative Deep Learning (2nd Edition)* explores various generative models, including GANs, VAEs, autoregressive models, diffusion models, Transformers, and more. This repository provides equivalent PyTorch implementations of the original Keras-based models.

---

## 🚀 Currently Implemented Models
*(More models to be added later)*

- **Chapter 2:**
  - MLP (Multilayer Perceptron)
  - CNN (Convolutional Neural Network)
- **Chapter 3:**
  - Autoencoder
  - VAE (Variational Autoencoder)
- **Chapter 4:**
  - GAN (Generative Adversarial Networks)
  - WGAN-GP (Wasserstein GAN with Gradient Penalty)
  - CGAN (Conditional GAN)
- **Chapter 5:**
  - LSTM (Long Short-Term Memory Network)
- **Chapter 6:**
  - RealNVP (Normalizing Flow Models)
- **Chapter 7:**
  - EBM (Energy-Based Models)
- **Chapter 8:**
  - DDM (Denoising Diffusion Models)
- **Chapter 9:**
  - GPT (Transformer Models)

---

## ⚙️ Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/generative_deep_Learning_2nd_edition_pytorch.git
   cd generative_deep_Learning_2nd_edition_pytorch
   ```

2. Build the Anaconda environment:
   ```bash
   conda env create -f requirements.yml
   ```

3. Activate the Anaconda environment:
   ```bash
   conda activate gen_dl_pytorch
   ```

4. Start the Jupyter notebook server:
   ```bash
   jupyter notebook
   ```

5. Download the dataset:
   
   Inside each notebook, check which dataset is required. Open a new terminal inside the repository and run the appropriate download script. For example, to download the CelebA dataset:
   ```bash
   bash scripts/download.sh faces
   ```

6. Run the notebook:
   
   Open the desired notebook in the Jupyter server and execute the cells.

7. Run the TensorBoard server:
   
   Launch TensorBoard for a specific experiment using:
   ```bash
   bash scripts/tensorboard.sh <chapter> <model_exp>
   ```
   Example for Conditional GAN:
   ```bash
   bash scripts/tensorboard.sh 04_gan 03_cgan
   ```
   Then open [http://localhost:6007](http://localhost:6007) in your browser to visualize.

---

## 🗂️ Repository Structure

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
│   ├── 06_normflow/
│   │   ├── 01_realnvp/
│   │   │   ├── realnvp_pytorch.ipynb
│   ├── 07_ebm/
│   │   ├── 01_ebm/
│   │   │   ├── ebm_pytorch.ipynb
│   ├── 08_diffusion/
│   │   ├── 01_ddm/
│   │   │   ├── ddm_pytorch.ipynb
│   ├── 09_transformer/
│   │   ├── 01_gpt/
│   │   │   ├── gpt_pytorch.ipynb
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
├── LICENCE               # Licence file
└── README.md             # This file
```

---
## 🔍 Differences from the Original Keras Implementation
- Fully re-implemented in PyTorch.
- Model architectures are intentionally kept as close as possible to the original.
- Training loops and utilities are adapted to PyTorch best practices.
- Some PyTorch-specific optimizations and improvements are included.

---

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

### 📌 Attribution

This project is an **unofficial PyTorch implementation** of examples from the book *Generative Deep Learning, 2nd Edition* by David Foster.

The original Keras companion code is available [here](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition) and is licensed under the Apache License 2.0.

---

## 🔗 References
- **Book:** *Generative Deep Learning (2nd Edition)* by David Foster
- **Original Code Repository:** [Keras Implementation](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main)

---

## 🤝 Contributing
Contributions are welcome! If you find any issues or would like to improve the implementations, feel free to submit a pull request.

---

## 📬 Contact
If you have any questions or suggestions, feel free to open an issue or contact:  
📧 **mary.raymond.n@gmail.com**
