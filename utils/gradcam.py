import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam


def create_heatmap_overlay(base_image, cam_map, alpha=0.32):
    if not isinstance(base_image, Image.Image):
        raise TypeError("base_image must be a PIL image.")

    image = base_image.convert("RGB").resize((224, 224))
    image_np = np.array(image).astype(np.float32) / 255.0

    normalized_map = np.clip(cam_map, 0.0, 1.0)
    softened_map = np.power(normalized_map, 0.85)
    heatmap_np = cm.get_cmap("turbo")(softened_map)[..., :3].astype(np.float32)
    overlay = np.clip((1.0 - alpha) * image_np + alpha * heatmap_np, 0.0, 1.0)
    return Image.fromarray((overlay * 255).astype(np.uint8))
