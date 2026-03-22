import io
import uuid
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

from model.classifier import load_classifier
from utils.gradcam import GradCAM, create_heatmap_overlay
from utils.preprocess import (
    classifier_tensor_from_pil,
    enhance_xray_for_display,
    ensure_runtime_directories,
    prepare_xray_image,
)

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
ENHANCED_DIR = OUTPUT_DIR / "enhanced"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
MODEL_DIR = BASE_DIR / "saved_models"
ensure_runtime_directories([UPLOAD_DIR, ENHANCED_DIR, HEATMAP_DIR, MODEL_DIR])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_PATH = MODEL_DIR / "classifier.pth"

_runtime_cache = {
    "classifier": None,
    "gradcam": None,
    "classifier_info": None,
}


def model_status():
    ready = CLASSIFIER_PATH.exists()
    return {
        "ready": ready,
        "status_text": "System Ready" if ready else "System Setup Required",
    }


def load_runtime_assets():
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError("The analysis system is not ready yet. Please set up the saved classifier first.")

    if _runtime_cache["classifier"] is None:
        classifier, target_layer, classifier_info = load_classifier(CLASSIFIER_PATH, DEVICE)
        _runtime_cache["classifier"] = classifier
        _runtime_cache["gradcam"] = GradCAM(classifier, target_layer)
        _runtime_cache["classifier_info"] = classifier_info

    return _runtime_cache["classifier"], _runtime_cache["gradcam"], _runtime_cache["classifier_info"]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"jpg", "jpeg", "png"}


def build_result_copy(predicted_index, confidence):
    confidence_percent = round(confidence * 100.0, 2)
    if predicted_index == 1:
        return {
            "label": "Pneumonia",
            "headline": "Possible Signs of Pneumonia Detected",
            "confidence": confidence_percent,
            "explanation": "This analysis suggests there may be changes linked to infection in the lungs. Clinical review is recommended.",
            "status_badge": "Follow-up Recommended",
        }

    return {
        "label": "Normal",
        "headline": "No Clear Signs of Pneumonia",
        "confidence": confidence_percent,
        "explanation": "This analysis suggests that the lungs appear normal. However, clinical confirmation is recommended.",
        "status_badge": "No Immediate Concern",
    }


@app.route("/")
def index():
    return render_template("index.html", status=model_status())


@app.route("/health")
def health():
    return jsonify(model_status())


@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    status = model_status()
    if not status["ready"]:
        return jsonify({"error": "The system is not ready yet. Please complete setup and try again."}), 503

    if "image" not in request.files:
        return jsonify({"error": "Please upload an X-ray image to continue."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Please choose an image file."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PNG and JPG images are supported."}), 400

    try:
        classifier, gradcam, classifier_info = load_runtime_assets()
        original_image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "We could not read that image. Please try another file."}), 400

    uid = uuid.uuid4().hex
    extension = file.filename.rsplit(".", 1)[1].lower()
    original_filename = f"{uid}_original.{extension}"
    original_path = UPLOAD_DIR / original_filename
    original_image.save(original_path)

    diagnosis_ready_image = prepare_xray_image(original_image)
    enhanced_image = enhance_xray_for_display(original_image)
    enhanced_filename = f"enhanced/{uid}_enhanced.png"
    enhanced_path = OUTPUT_DIR / enhanced_filename
    enhanced_image.save(enhanced_path)

    try:
        classifier_input = classifier_tensor_from_pil(diagnosis_ready_image, DEVICE)
        with torch.no_grad():
            logits = classifier(classifier_input)
            probabilities = torch.softmax(logits, dim=1)[0]

        pneumonia_probability = float(probabilities[1].item())
        threshold = float(classifier_info.get("threshold", 0.5))
        predicted_index = 1 if pneumonia_probability >= threshold else 0
        confidence = pneumonia_probability if predicted_index == 1 else (1.0 - pneumonia_probability)

        cam_map = gradcam.generate(classifier_input, target_class=predicted_index)
        heatmap_overlay = create_heatmap_overlay(diagnosis_ready_image, cam_map)
        heatmap_filename = f"heatmaps/{uid}_heatmap.png"
        heatmap_path = OUTPUT_DIR / heatmap_filename
        heatmap_overlay.save(heatmap_path)
    except Exception:
        return jsonify({"error": "The image could not be analyzed right now. Please try again."}), 500

    result_copy = build_result_copy(predicted_index, confidence)
    return jsonify(
        {
            "result": result_copy["headline"],
            "label": result_copy["label"],
            "confidence": result_copy["confidence"],
            "explanation": result_copy["explanation"],
            "status_badge": result_copy["status_badge"],
            "original_image": f"/uploads/{original_filename}",
            "enhanced_image": f"/outputs/{enhanced_filename}",
            "heatmap_image": f"/outputs/{heatmap_filename}",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
