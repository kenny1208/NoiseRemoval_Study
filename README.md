# A Comparative Study of DnCNN and Traditional Filters for Multi-Type Image Noise Removal

This repository contains the code and resources for my research project:  
**"A Comparative Study of DnCNN and Traditional Filters for Multi-Type Image Noise Removal."**  
The accompanying paper (included as a PDF in this repo) presents a systematic comparison between the deep learning model **DnCNN** and traditional filtering techniques (Median, Gaussian, and Low-Pass) for denoising images corrupted by multiple noise types.

---

## 📄 Paper

The full paper is available here:  
[`A Comparative Study of DnCNN and Traditional Filters for.pdf`](./A%20Comparative%20Study%20of%20DnCNN%20and%20Traditional%20Filters%20for.pdf)

---

## 📂 Repository Structure

```
.
├── data_generator.py     # Generates image patches, noise augmentation, dataset utilities
├── main_train.py         # PyTorch implementation of DnCNN model and training pipeline
├── metric.ipynb          # Jupyter notebook for evaluation metrics and result analysis
├── addnoise.cpp          # Adds Gaussian, Salt-and-Pepper, and Random noise to grayscale images
├── denoise.cpp           # Implements Median, Gaussian, and Low-Pass filters in C++
├── A Comparative Study of DnCNN and Traditional Filters for.pdf  # Research paper
```

---

## 🚀 Features

- **DnCNN Deep Learning Model**

  - Residual learning CNN for denoising.
  - Mixed loss function (Charbonnier + MSE).
  - Dynamic noise-ratio training strategy for robustness.

- **Traditional Filters (C++ Implementations)**

  - Median Filter: effective against Salt-and-Pepper noise.
  - Gaussian Filter: suppresses Gaussian noise.
  - Low-Pass Filter: averages neighborhood pixels but may blur edges.

- **Noise Generation**

  - Tools to add Gaussian, Salt-and-Pepper, and Random noise to grayscale BMP images.

- **Evaluation**
  - Metrics include PSNR, SSIM, and MSE.
  - Jupyter notebook (`metric.ipynb`) for analysis and visualization.

---

## 📈 Experimental Results

- **DnCNN** performed best on Gaussian and Random noise (highest PSNR & SSIM, lowest MSE).
- **Median Filter** outperformed DnCNN on Salt-and-Pepper noise, showing classical filters still have value in specific scenarios.

---

## 📬 Contact

- **Author**: 陳禹豪 (Yu-Hao Chen)
- **Email**: 1160801@mcu.edu.tw
- **Affiliation**: Ming Chuan University, Taiwan

---
