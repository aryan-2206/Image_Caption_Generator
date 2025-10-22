# Image Captioning Generator

An end-to-end deep learning project that generates descriptive captions for images using an encoder-decoder architecture with attention mechanisms.

---

## Project Overview

This project implements an image captioning system that can generate human-like captions for images. It uses:

- **Encoder**: Convolutional Neural Network (CNN) to extract image features.
- **Decoder**: Recurrent Neural Network (RNN) or LSTM with attention to generate captions.
- **Attention Mechanism**: Highlights relevant parts of the image for each word in the caption.

The project also supports evaluation using BLEU scores and visualization of attention heatmaps.

---

## Features

- Trainable end-to-end image captioning model.
- Generates captions for new images.
- Attention visualization to understand model focus.
- Evaluation using BLEU metrics.
- Modular code for easy experimentation.

---

## Project Structure

```text
image-captioning/
│
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── images/          # Raw image dataset
│   ├── captions/        # Caption files (JSON, CSV, or txt)
│   └── processed/       # Preprocessed data, tokenized captions
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Source code
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── visualize.py
├── checkpoints/         # Saved model weights
└── outputs/             # Generated captions, plots, evaluation reports

```

## Installation

Follow these steps to set up the project:

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/image-captioning.git
cd image-captioning

2. **Create and activate a virtual environment:**


python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate

3. **Install dependencies:**

pip install -r requirements.txt

## Usage

1. **Train the model:**

- un the training script with your dataset path, number of epochs, and batch size:

- python src/train.py --data_path data/ --epochs 20 --batch_size 32

2. **Generate captions for a single image:**

- Use the evaluation script with an image and a trained checkpoint:

- python src/evaluate.py --image_path data/images/sample.jpg --checkpoint checkpoints/best_model.pth

3. **Visualize attention maps:**

- Visualize which parts of the image the model focuses on while generating a caption:

- python src/visualize.py --image_path data/images/sample.jpg --caption "a man riding a snowboard" --alphas alphas.npy

## Evaluation Metrics

- BLEU-1 to BLEU-4: Measures n-gram overlap between generated and reference captions.
- Optionally, you can also include: METEOR, ROUGE, CIDEr for more comprehensive evaluation.

These metrics give an idea of how accurate and human-like your generated captions are compared to reference captions.

## Results

Include sample generated captions, attention visualizations, and evaluation scores. Example:

Image: data/images/sample.jpg

Generated Caption: "A man riding a snowboard down a snowy slope"

Attention Heatmap: outputs/attention/sample.png

BLEU Scores:

BLEU-1: 0.39
BLEU-4: 0.13