# 🎭 Facial Emotion Recognition (FER) using CNN

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX_1060-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com/)

An end-to-end Deep Learning pipeline to classify human facial expressions into 7 categories. This project leverages a custom Convolutional Neural Network (CNN) architecture and was trained on a balanced version of the RAF-DB dataset, achieving an impressive **90% validation accuracy**.

---

## 🚀 Overview
The goal of this project is to recognize emotions from grayscale facial images. By utilizing spatial feature extraction through multiple convolutional blocks, the model learns to identify key "micro-expressions" such as the curve of a smile or the furrow of a brow.

### Key Features
*   **7-Class Classification:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
*   **Optimized Architecture:** Includes Batch Normalization and Dropout for high stability and generalization.
*   **High Performance:** Optimized for real-time inference on mid-range GPUs (like the NVIDIA RTX 1060).
*   **Preprocessing Pipeline:** Automatic grayscale conversion and $75 \times 75$ pixel normalization.

---

## 🧠 Model Architecture
The model consists of 4 main Convolutional Blocks followed by Fully Connected layers. Each block is designed to increase feature depth while reducing spatial dimensions.

| Layer | Input Size | Output Channels | Operation |
| :--- | :--- | :--- | :--- |
| **Conv Block 1** | $75 \times 75 \times 1$ | 32 | $3 \times 3$ Conv, BN, ReLU, MaxPool |
| **Conv Block 2** | $37 \times 37 \times 32$ | 64 | $3 \times 3$ Conv, BN, ReLU, MaxPool |
| **Conv Block 3** | $18 \times 18 \times 64$ | 128 | $3 \times 3$ Conv, BN, ReLU, MaxPool |
| **Conv Block 4** | $9 \times 9 \times 128$ | 256 | $3 \times 3$ Conv, BN, ReLU, MaxPool |
| **Dense 1** | $4 \times 4 \times 256$ (4096) | 512 | Linear, ReLU, Dropout (0.5) |
| **Output** | 512 | 7 | Linear (Logits) |

---

## 📊 Dataset & Training
The project uses a balanced version of the **RAF-DB (Real-world Affective Faces Database)**.

*   **Total Images:** ~62,000 (after augmentation and class balancing).
*   **Balancing Strategy:** Oversampling and data augmentation were applied to prevent majority-class bias (e.g., toward "Happy").
*   **Training Platform:** Kaggle (Tesla P100/T4 GPU).
*   **Final Performance:** **90% Accuracy** on the validation set.

---

## 💻 Installation & Usage

### 1. Requirements
*   Python 3.10+
*   PyTorch & Torchvision
*   Pillow
*   CUDA (Highly recommended for RTX 1060 users)

### 2. Local Setup
```bash
# Clone the repository
git clone [https://github.com/yourusername/emotion-detection-cnn.git](https://github.com/yourusername/emotion-detection-cnn.git)
cd emotion-detection-cnn

# Install dependencies
pip install torch torchvision pillow
