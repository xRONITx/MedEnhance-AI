from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

ENHANCER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

CLASSIFIER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def ensure_runtime_directories(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_pil_image(image_or_path):
    if isinstance(image_or_path, (str, Path)):
        return Image.open(image_or_path).convert("RGB")
    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")
    raise TypeError("Expected a PIL image or a filesystem path.")


def prepare_xray_image(image_or_path):
    image = load_pil_image(image_or_path)
    return image.convert("L").convert("RGB")


def enhance_xray_for_display(image_or_path):
    image = prepare_xray_image(image_or_path).convert("L")
    image_np = np.array(image)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast_boosted = clahe.apply(image_np)

    blurred = cv2.GaussianBlur(contrast_boosted, (0, 0), sigmaX=1.1)
    sharpened = cv2.addWeighted(contrast_boosted, 1.55, blurred, -0.55, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened).convert("RGB")


def enhancer_tensor_from_pil(image_or_path, device=None):
    image = prepare_xray_image(image_or_path)
    tensor = ENHANCER_TRANSFORM(image).unsqueeze(0)
    return tensor.to(device) if device is not None else tensor


def classifier_tensor_from_pil(image_or_path, device=None):
    image = prepare_xray_image(image_or_path)
    tensor = CLASSIFIER_TRANSFORM(image).unsqueeze(0)
    return tensor.to(device) if device is not None else tensor


def preprocess_image(image_or_path, device=torch.device("cpu")):
    return classifier_tensor_from_pil(image_or_path, device=device)


def pil_from_tensor(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array, 0.0, 1.0)
    return Image.fromarray((array * 255).astype(np.uint8))


def degrade_tensor(clean_tensor, noise_std_range=(0.03, 0.09), min_scale=0.45):
    if clean_tensor.dim() != 3:
        raise ValueError("degrade_tensor expects a 3D CHW tensor.")

    clean_tensor = clean_tensor.clamp(0.0, 1.0)
    noise_std = torch.empty(1).uniform_(*noise_std_range).item()
    noisy = (clean_tensor + torch.randn_like(clean_tensor) * noise_std).clamp(0.0, 1.0)

    _, height, width = noisy.shape
    scale = torch.empty(1).uniform_(min_scale, 0.8).item()
    reduced_h = max(64, int(height * scale))
    reduced_w = max(64, int(width * scale))

    reduced = torch.nn.functional.interpolate(
        noisy.unsqueeze(0),
        size=(reduced_h, reduced_w),
        mode="bilinear",
        align_corners=False,
    )
    restored = torch.nn.functional.interpolate(
        reduced,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return restored.clamp(0.0, 1.0)


def find_dataset_root(project_dir):
    project_dir = Path(project_dir).resolve()
    candidates = [
        project_dir / "chest_xray",
        project_dir.parent / "chest_xray",
        project_dir.parent / "chest_xray" / "chest_xray",
        project_dir / "data" / "chest_xray",
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate chest_xray dataset. Checked: " + ", ".join(str(path) for path in candidates)
    )


def list_image_files(root_dir):
    root_dir = Path(root_dir)
    return sorted(
        path for path in root_dir.rglob("*") if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )
