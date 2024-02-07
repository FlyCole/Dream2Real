import pdb
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from pdb import set_trace
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
import numpy as np

import torch

def txt2img(prompt):
    pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
    pipeline.to("cuda")

    images = pipeline(prompt=prompt).images
    return images

# For mask, white pixels will be inpainted, black pixels will be preserved.
# C, H, W format for image. Just H and W for mask.
def inpaint(img, mask=None, pipeline=None, prompt=""):
    orig_size = img.size()
    img = resize(img, (512, 512), interpolation=InterpolationMode.NEAREST)
    # If no mask is provided, we will use the white pixels as the mask.
    if mask is None:
        img_tensor = pil_to_tensor(img).to("cuda")
        mask = torch.where(torch.all(img_tensor > 200, dim=0), 255, 0)
        mask_image = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
    else:
        # Format hell.
        img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        mask_image = resize(mask.unsqueeze(0).repeat(orig_size[0], 1, 1), (512, 512), interpolation=InterpolationMode.NEAREST)
        mask_image = Image.fromarray(np.logical_not(mask_image.permute(1, 2, 0).cpu().numpy()).astype(np.uint8) * 255)

    if pipeline is None:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", requires_safety_checker=False)
        pipeline.to("cuda")

    out_img = pipeline(prompt=prompt, image=img, mask_image=mask_image).images[0]
    out_img = out_img.resize(orig_size[1:], Image.NEAREST)
    return out_img

if __name__ == "__main__":
    img = Image.open("data/inpainting/pcd_render_square.png")
    inpainted = inpaint(img)
    inpainted[0].show()
