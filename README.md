# Audio Deepfake Detection - ESDD2 Challenge

This repository contains the codebase to train a 5-class audio deepfake detection model specifically for the "ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge" (Zhang et al., Jan 2026), working with the CompSpoofV2 dataset.

The 5-class problem generally tackles the permutations of genuine/spoofed speech and genuine/spoofed environmental sounds.

## Installation

1. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset (CompSpoofV2)

Place the dataset locally inside a `data/CompSpoofV2` folder. The structure expected is:
```text
data/CompSpoofV2/
├── train/
│   ├── class_0/
│   ├── class_1/
│   ├── ...
├── dev/
├── eval/
```
*(Alternatively, you can provide a CSV file mapping audio files to one of the 5 labels.)*

## Usage

### 1. Preprocess and Train
Run the training script once your dataset is downloaded.
```bash
python train.py --data_dir ./data/CompSpoofV2 --epochs 20 --batch_size 32
```

### 2. Evaluation
To evaluate your checkpoint on the validation/eval set:
```bash
python evaluate.py --checkpoint models/best_model.pth --data_dir ./data/CompSpoofV2
```
