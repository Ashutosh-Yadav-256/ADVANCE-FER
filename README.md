# Advanced Facial Expression Recognition (FER) System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A state-of-the-art Facial Expression Recognition system leveraging **PyTorch**, **MediaPipe**, and **Deep Learning** to detect emotions in real-time. This project combines robust visual feature extraction with geometric landmark analysis for high-accuracy emotion classification.

## Features

- **Hybrid Architecture**: Combines CNN backbones (ResNet/EfficientNet) with geometric features from MediaPipe landmarks.
- **Robust Feature Extraction**: Utilizes 468 3D facial landmarks for precise geometric understanding.
- **Real-time Inference**: Optimized pipeline capable of running on CPU/GPU for live webcam feeds.
- **Explainability**: (Planned) Integration of Grad-CAM for model decision visualization.
- **Easy Deployment**: Streamlit-based web interface for immediate testing and demonstration.

## Directory Structure

```
ADVANCE FER/
├── data/               # Dataset storage (FER2013)
├── models/             # Model definitions and saved weights
│   ├── fer_model.py    # Main model architecture
│   └── ...
├── deployment/         # Inference logic
├── utils/              # Helper scripts (visualization, etc.)
├── app.py              # Streamlit demo application
├── train.py            # Training script
├── download_data.py    # Dataset download script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation

1.  **Clone the repository** (if applicable) or download the source code.

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Setup

This project uses the **FER2013** dataset. You can automatically download it using the provided script:

```bash
python download_data.py
```

*Note: You may need a Kaggle account and API key configured if the script uses the Kaggle API.*

## Training

To train the model from scratch:

```bash
python train.py
```

Key hyperparameters (batch size, learning rate, epochs) can be modified directly in `train.py`.

## Usage / Demo

Run the interactive Streamlit application to test the model with your webcam or upload images/videos:

```bash
python -m streamlit run app.py
```

## Model Architecture

The model uses a dual-stream approach:
1.  **Visual Stream**: A CNN backbone (ResNet50 or EfficientNet) extracts texture and shape information from the raw image.
2.  **Geometric Stream**: A Landmark Encoder processes 468 3D facial landmarks extracted by MediaPipe.
3.  **Fusion**: Features from both streams are concatenated and passed through a classification head to predict one of 7 emotions: *Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MediaPipe** for the robust Face Mesh solution.
- **PyTorch** for the deep learning framework.
- **FER2013** dataset creators.
