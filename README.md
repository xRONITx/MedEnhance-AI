# MedEnhance AI

MedEnhance AI is an AI-assisted chest X-ray analysis web application built to support medical image review with a clean clinical workflow. The system combines image enhancement, pneumonia screening, confidence reporting, and visual explanation into a single professional web interface.

It is designed as an academic and demonstration-ready healthcare AI project, with a modular training pipeline and an explainable inference workflow that helps users understand what influenced the model's decision.

## Highlights

- Chest X-ray upload and guided analysis workflow
- Medical image enhancement for clearer visual review
- Pneumonia detection using a ResNet18-based classifier
- Grad-CAM heatmap visualization for explainability
- Confidence scoring and diagnosis summary
- Professional single-page medical interface
- Modular training scripts for classifier and enhancer
- Flask-based deployment-ready application structure

## Tech Stack

- Python
- Flask
- PyTorch
- Torchvision
- NumPy
- Pillow
- scikit-image
- scikit-learn
- HTML
- CSS
- JavaScript

## Project Structure

```text
project/
|-- app.py
|-- training/
|   |-- train_classifier.py
|   `-- train_enhancer.py
|-- model/
|   |-- classifier.py
|   `-- enhancer.py
|-- utils/
|   |-- gradcam.py
|   |-- metrics.py
|   `-- preprocess.py
|-- static/
|   |-- script.js
|   `-- styles.css
|-- templates/
|   `-- index.html
|-- uploads/
|-- outputs/
|   |-- enhanced/
|   `-- heatmaps/
|-- saved_models/
|   |-- classifier.pth
|   |-- classifier_metrics.json
|   |-- enhancer.pth
|   `-- enhancer_metrics.json
|-- requirements.txt
`-- README.md
```

## Dataset

The project expects the chest X-ray dataset in one of these layouts:

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

The training scripts automatically detect both structures.

## Features

### 1. Image Enhancement
The project includes a dedicated enhancement model trained on degraded chest X-rays. During training, images are artificially corrupted using noise and resolution degradation so the model learns to reconstruct cleaner outputs.

### 2. Pneumonia Detection
The classifier is built on ResNet18 and fine-tuned for binary classification:
- Normal
- Pneumonia

The improved training pipeline includes:
- balanced sampling for class imbalance
- stronger augmentation
- validation split refinement
- confidence threshold tuning
- metric tracking for retraining analysis

### 3. Explainable AI
The application generates a Grad-CAM heatmap that highlights the regions of the X-ray that most influenced the prediction.

### 4. Clinical Web Workflow
The web app follows a simple review flow:
1. Upload image
2. Preview original X-ray
3. Start analysis
4. View enhanced image, heatmap, diagnosis, confidence, and explanation

## Setup

### 1. Create a virtual environment
```bash
python -m venv venv
```

### 2. Activate the virtual environment
On Windows:
```bash
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Training

### Train the enhancer
```bash
python training/train_enhancer.py
```

Optional example:
```bash
python training/train_enhancer.py --epochs 14 --batch-size 12 --learning-rate 0.0007
```

### Train the classifier
```bash
python training/train_classifier.py
```

Optional example:
```bash
python training/train_classifier.py --epochs 16 --batch-size 20 --learning-rate 0.0003
```

## Running the Application

After the required models are available:

```bash
python app.py
```

Open the app in your browser:

```text
http://127.0.0.1:5000
```

## Inference Workflow

When a user uploads an X-ray, the application performs the following steps:

1. Loads the uploaded image
2. Shows the original image preview
3. Enhances the image for clearer visual review
4. Runs the trained classifier
5. Generates a Grad-CAM heatmap
6. Displays:
   - original image
   - enhanced image
   - heatmap
   - diagnosis result
   - confidence level
   - explanation text

## Metrics Tracked

### Classification
- Accuracy
- Precision
- Recall
- F1 Score

### Enhancement
- PSNR
- SSIM

Training summaries are stored in:
- `saved_models/classifier_metrics.json`
- `saved_models/enhancer_metrics.json`

## Notes

- The application is intended as an AI assistance tool, not as a replacement for professional diagnosis.
- Generated files in `uploads/` and `outputs/` are runtime artifacts and are not required for source control.
- Model weight files can be retrained using the included training scripts.

## Recommended Usage Order

1. Install dependencies
2. Train the enhancer
3. Train the classifier
4. Run the Flask app
5. Test with chest X-ray images

## Future Improvements

- Better calibration and threshold tuning on larger validation sets
- Multi-condition classification support
- Cloud deployment
- User authentication and patient workflow integration
- Exportable medical report summaries

## Disclaimer

This project is for educational, research, and demonstration purposes only. It should not be used as a standalone clinical diagnostic system.
