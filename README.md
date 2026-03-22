# MedEnhance AI

MedEnhance AI is a Flask-based medical image analysis system for chest X-rays. It includes:

- A CNN enhancement model trained on artificially degraded X-rays
- A binary pneumonia classifier built on pretrained ResNet18
- Real Grad-CAM visualization on the classifier's last convolutional layer
- A polished drag-and-drop web interface for end-to-end inference

## Project Structure

```text
project/
|-- app.py
|-- training/
|   |-- train_classifier.py
|   |-- train_enhancer.py
|   `-- train_model.py
|-- model/
|   |-- classifier.py
|   |-- enhancer.py
|   `-- model.py
|-- utils/
|   |-- preprocess.py
|   |-- gradcam.py
|   `-- metrics.py
|-- static/
|   |-- styles.css
|   `-- script.js
|-- templates/
|   `-- index.html
|-- uploads/
|-- outputs/
|   |-- enhanced/
|   `-- heatmaps/
|-- saved_models/
|   |-- classifier.pth
|   `-- enhancer.pth
|-- requirements.txt
`-- README.md
```

## Dataset

The training scripts auto-detect either of these layouts:

```text
chest_xray/
|-- train/
|-- val/
`-- test/
```

or

```text
chest_xray/
`-- chest_xray/
    |-- train/
    |-- val/
    `-- test/
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train The Enhancer

```bash
python training/train_enhancer.py
```

Optional example:

```bash
python training/train_enhancer.py --epochs 12 --batch-size 12 --learning-rate 0.001
```

## Train The Classifier

```bash
python training/train_classifier.py
```

Optional example:

```bash
python training/train_classifier.py --epochs 8 --batch-size 16
```

Notes:

- `train_classifier.py` uses pretrained ResNet18 by default.
- Torchvision may download ImageNet weights on the first run if they are not cached.
- Best checkpoints are saved to `saved_models/classifier.pth` and `saved_models/enhancer.pth`.
- Metric summaries are written to `saved_models/classifier_metrics.json` and `saved_models/enhancer_metrics.json`.

## Run The Web App

After both models are trained:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Inference Pipeline

1. Upload chest X-ray
2. Enhance image with the trained U-Net enhancer
3. Classify the enhanced image with ResNet18
4. Generate Grad-CAM heatmap
5. Display original image, enhanced image, heatmap, prediction, and confidence

## Metrics

The project computes:

- Accuracy
- Precision
- Recall
- F1 Score
- PSNR
- SSIM

## Recommended Run Order

1. `python training/train_enhancer.py`
2. `python training/train_classifier.py`
3. `python app.py`

If the app starts before training, it will still render the interface and report which model files are missing.
