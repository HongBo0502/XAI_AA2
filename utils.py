import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import copy
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.target_layer = self.get_target_layer()
        self.hook()

    def get_target_layer(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module
        raise ValueError(f"Layer {self.target_layer_name} not found!")

    def hook(self):
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)
        class_score = output
        self.model.zero_grad()
        class_score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0).detach().cpu().numpy()
        heatmap = heatmap / np.max(heatmap)  # Normalize heatmap
        return heatmap

    def visualize_heatmap(self, heatmap, image, alpha=0.5,save_path=None):
        heatmap = resize(heatmap, (image.size[1], image.size[0]))
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = plt.cm.jet(heatmap)[:, :, :3]  # Apply colormap
        overlay = np.array(image) * alpha + heatmap * (1 - alpha)
        overlay = overlay.astype(np.uint8)

        # Save the overlay image if save_path is provided
        if save_path:
            overlay_image = Image.fromarray(overlay)
            overlay_image.save(save_path)
            print(f"Grad-CAM overlay saved to: {save_path}")
    
        return overlay
    
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def visualize_gradcam(image_tensor, heatmap, alpha=0.5, colormap='jet', normalize=True,save_path=None):
    """
    Visualizes the Grad-CAM heatmap overlayed on the input image.
    
    Parameters:
    - image_tensor: The input image tensor of shape (B, C, H, W) or (C, H, W).
    - heatmap: The Grad-CAM heatmap of shape (H, W).
    - alpha: The blending factor between the image and the heatmap (default 0.5).
    - colormap: The colormap for the heatmap (default 'jet').
    - normalize: Whether to normalize the image to [-1, 1] (default True).
    """
    # If the image tensor has a batch dimension, squeeze it (take the first image)
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)  # Remove batch dimension

    # Convert image tensor to numpy (C, H, W) format for processing
    image = image_tensor.permute(1, 2, 0).detach().cpu().numpy()

    # Normalize the image to [-1, 1] or to [0, 1]
    if normalize:
        image = 2 * ((image - image.min()) / (image.max() - image.min())) - 1  # Normalize to [-1, 1]
    else:
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]

    # Resize the heatmap to match the image dimensions
    heatmap_resized = resize(heatmap, (image.shape[0], image.shape[1]), mode='reflect')

    # Apply the selected colormap to the heatmap and blend it with the original image
    heatmap_overlay = (alpha * plt.cm.jet(heatmap_resized)[:, :, :3] + 
                       (1 - alpha) * (image + 1) / 2)  # Adjust for blending

    # Clip the values to [0, 1] for proper display
    heatmap_overlay = np.clip(heatmap_overlay, 0, 1)
    

    # Save the heatmap_overlay if save_path is provided
    if save_path:
        plt.imsave(save_path, heatmap_overlay)
        print(f"Heatmap overlay saved to {save_path}")
def generate_image_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor
import torch
from torchvision.models import densenet121
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from PIL import Image
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlusFlat, EpsilonGammaBox
from zennit.types import BatchNorm
from zennit.image import imgify
from zennit.rules import Epsilon
from torchvision.transforms import ToPILImage
def lrp_image(image_path,image_tensor,model,save_path=None):
    # Load original image
    ori_Image = Image.open(image_path).resize((224, 224))

    # Perform LRP
    composite = EpsilonGammaBox(low=-3.0, high=3.0, layer_map=[(BatchNorm, Epsilon())])
    input = image_tensor.clone().requires_grad_(True)
    target = torch.tensor([[1.0]], device=device)  # Adjusted for binary classification (use 1.0 for the target class)

    # Perform LRP attribution
    with Gradient(model, composite) as attributor:
        output, relevance = attributor(input, target)

    # Convert relevance to an image (heatmap)
    relevance_heatmap = relevance[0].detach().sum(0)
    relevance = imgify(relevance_heatmap.cpu(), cmap='bwr', symmetric=True)

    # Create an alpha mask
    alpha = 0.4
    alpha_mask = Image.new("L", ori_Image.size, int(alpha * 255))  # Grayscale mask where alpha defines transparency

    # Create the mask image with transparent RGB channels and apply the alpha mask
    mask_img = ori_Image.convert("RGB")  # Copy the original image
    mask_img.putalpha(alpha_mask)  # Apply the alpha mask for transparency

    # Overlay the relevance (heatmap) on the original image
    overlay = Image.alpha_composite(relevance.convert("RGBA"), mask_img)


    # Optionally save the overlay image
    overlay.save(save_path)



def generate_single_perturbed_image(image, patch_ratio=0.5, replacement_value=0, offset_x=0, offset_y=0):
    """
    Generate a single perturbed image by masking a region of the input image.
    
    Args:
        image (PIL.Image): The input image to perturb.
        patch_ratio (float): Ratio (0 < patch_ratio <= 1) of the image to be masked.
                             A larger value masks more of the image.
        replacement_value (int): Pixel value to fill the masked region (0 for black, 255 for white, etc.).
        offset_x (int): Horizontal offset for moving the patch (-ve for left, +ve for right).
        offset_y (int): Vertical offset for moving the patch (-ve for up, +ve for down).
    
    Returns:
        PIL.Image: The perturbed image with the masked region.
    """
    original = np.array(image)
    h, w, _ = original.shape

    # Calculate patch size based on ratio
    patch_width = int(w * patch_ratio)
    patch_height = int(h * patch_ratio)

    # Define the patch's starting coordinates with offsets
    start_x = max(0, min((w - patch_width) // 2 + offset_x, w - patch_width))
    start_y = max(0, min((h - patch_height) // 2 + offset_y, h - patch_height))

    # Create a copy and mask the specified patch
    perturbed = original.copy()
    perturbed[start_y:start_y + patch_height, start_x:start_x + patch_width, :] = replacement_value

    return Image.fromarray(perturbed)

