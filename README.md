# LIP READING : Speech Recognition without audio using 3D Convolutional Neural Networks
## Project Overview
A deep learning-based audio speech recognition application. This project demonstrates lip reading using a 3D CNN model which analyzes lip motion from face video and outputs the predicted text without using audio.

**Key Capabilities:**
- Audio speech recognition using deep learning
- Real-time inference on face video
- User-friendly Streamlit interface

---

## Dataset

**GRID Corpus - Speaker S1**

The GRID (Grid Audiovisual Sentence) corpus is a multitalker audiovisual sentence corpus for speech perception research.

**Dataset Specifications:**
- **Source:** GRID Corpus
- **Speaker Used:** S1 (subset)
  - Example: "bin blue at G9 now"
- **Download:** [GRID Corpus Official Site](http://spandh.dcs.shef.ac.uk/gridcorpus/)

---

## Installation

### Prerequisites

```bash
Python 3.8+
pip
Git
```

### Step-by-Step Setup

1. **Clone the repository**

2. **Create virtual environment**

3. **Activate virtual environment**

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Application

1. **Start Streamlit app**
```bash
streamlit run app.py
```

2. **Access the application**
- Open browser at `http://localhost:8501`

---

## Model Architecture

**Neural Network Design:**

The model is a 3D-CNN + BiLSTM network implemented in Keras. Input shape: (75, 46, 140, 1) — 75 frames of 46x140 grayscale crop.

Layer sequence:
- Conv3D(128, kernel_size=3, padding='same') + ReLU
- MaxPool3D(pool_size=(1,2,2))
- Conv3D(256, kernel_size=3, padding='same') + ReLU
- MaxPool3D(pool_size=(1,2,2))
- Conv3D(75, kernel_size=3, padding='same') + ReLU
- MaxPool3D(pool_size=(1,2,2))
- TimeDistributed(Flatten())
- Bidirectional(LSTM(128, return_sequences=True)) + Dropout(0.5)
- Bidirectional(LSTM(128, return_sequences=True)) + Dropout(0.5)
- Dense(vocab_size + 1, activation='softmax')  — final per-timestep character probabilities (CTC training)

**Model Summary:**
```
 Total params: 8,471,924 (32.32 MB)
 Trainable params: 8,471,924 (32.32 MB)
 Non-trainable params: 0 (0.00 B)
```

---

## Model Performance

**Evaluation Metrics on GRID Corpus (S1 Test Split):**

| Metric | Value |
|--------|-------|
| Word Error Rate (WER) | 0.0050 (0.50%) |
| Character Error Rate (CER) | 0.0016 (0.16%) |
| Model Weights | checkpoint.weights.h5 |
| Dataset | GRID Corpus - Speaker S1 |

These metrics demonstrate the model's high accuracy in predicting spoken text from video alone, with less than 1% word error rate and 0.16% character error rate.

---

## Demo Video

Check out the demo video to see the lip reading model in action:

[![Lip Reading Demo](https://img.shields.io/badge/Demo-Video-blue?style=for-the-badge)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

*Note: Replace the YouTube video link with your actual demo video URL*

The demo video shows:
- Real-time lip reading inference on video input
- Text output prediction from mouth movements
- Model accuracy on various speech samples

---

## Live Demo

You can try the live deployed app here:

**[Streamlit Live App](https://lipreading-using3dcnn.streamlit.app/)**

Open the link in a browser to interact with the Streamlit interface and test the model with your own videos.
