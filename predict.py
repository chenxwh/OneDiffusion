# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from cog import BasePredictor, Input, Path
import torch
import matplotlib
from PIL import Image
import numpy as np
from onediffusion.diffusion.pipelines.onediffusion import OneDiffusionPipeline


MODEL_CACHE = "model_cache"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/lehduong/OneDiffusion/model_cache.tar"
)

TASKS = [
    "text2image",
    "deblurring",
    "image_inpainting",
    "canny2image",
    "depth2image",
    "hed2img",
    "pose2image",
    "semanticmap2image",
    "boundingbox2image",
    "image_editing",
    "faceid",
    "multiview",
    "subject_driven",
]
NEGATIVE_PROMPT = "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            print("downloading")
            download_weights(MODEL_URL, MODEL_CACHE)

        device = torch.device("cuda:0")
        self.pipe = OneDiffusionPipeline.from_pretrained(
            f"{MODEL_CACHE}/lehduong/OneDiffusion"
        ).to(device=device, dtype=torch.bfloat16)

    def predict(
        self,
        task: str = Input(
            description="Choose a task",
            choices=TASKS,
            default="text2image",
        ),
        prompt: str = Input(
            description="Input prompt.",
            default="A bipedal black cat wearing a huge oversized witch hat, a wizards robe, casting a spell,in an enchanted forest. The scene is filled with fireflies and moss on surrounding rocks and trees",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=NEGATIVE_PROMPT,
        ),
        image1: Path = Input(
            description="First input image for img2img tasks", default=None
        ),
        image2: Path = Input(
            description="Optional, second input image for img2img tasks", default=None
        ),
        image3: Path = Input(
            description="Optinal, third input image for img2img tasks", default=None
        ),
        use_input_image_size: bool = Input(
            description="Set the dimension of the output image the same as the input image",
            default=False,
        ),
        width: int = Input(
            description="Width of output image. Ignored when use_input_image_size is set to True. Multiview generation only supporst SQUARE image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image. Ignored when use_input_image_size is set to True. Multiview generation only supporst SQUARE image",
            default=1024,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=4
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        denoise_mask: str = Input(
            description="Denoise mask for output images, comma-separated 0s or/and 1s",
            default="0",
        ),
        azimuth: str = Input(
            description="Azimuths degrees, comma-separated, for multiview generation",
            default="0",
        ),
        elevation: str = Input(
            description="Elevations degrees, comma-separated, for multiview generation",
            default="0",
        ),
        distance: str = Input(
            description="Distances, comma-separated, for multiview generation",
            default="1.5",
        ),
        focal_length: float = Input(
            description="Focal Length of camera for multiview generation",
            default=1.3887,
        ),
    ) -> list[Path]:
        """Run a single prediction on the model"""

        images = [
            Image.open(str(img)).convert("RGB")
            for img in [image1, image2, image3]
            if img
        ]

        if not task == "text2image":
            assert len(images) > 0, f"Please provide input image for the {task} task."

        if use_input_image_size and len(images) > 0:
            width, height = images[0].size

        width, height = 16 * round(width / 16), 16 * round(height / 16)
        print(f"Using width, height: {width}, {height}")

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)
        scale_factor, scale_watershed, noise_scale = 1, 1, 1

        prompt = f"[[{task}]] {prompt}"
        print(f"Using prompt: {prompt}")

        if task == "text2image":
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                scale_factor=scale_factor,
                scale_watershed=scale_watershed,
                noise_scale=noise_scale,
            )
        else:
            denoise_mask = [int(d.strip()) for d in denoise_mask.split(",")]
            img2img_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "forward_kwargs": {
                    "scale_factor": scale_factor,
                    "scale_watershed": scale_watershed,
                },
                "noise_scale": noise_scale,
                "generator": generator,
                "denoise_mask": denoise_mask,
            }
            if task == "multiview":
                width, height = min(width, height), min(width, height)
                denoise_mask = [f"image_{d}" for d in denoise_mask]
                # Parse azimuth, elevation, and distance into lists, allowing 'None' values
                azimuths = (
                    [
                        float(a.strip()) if a.strip().lower() != "none" else None
                        for a in azimuth.split(",")
                    ]
                    if azimuth
                    else []
                )
                elevations = (
                    [
                        float(e.strip()) if e.strip().lower() != "none" else None
                        for e in elevation.split(",")
                    ]
                    if elevation
                    else []
                )
                distances = (
                    [
                        float(d.strip()) if d.strip().lower() != "none" else None
                        for d in distance.split(",")
                    ]
                    if distance
                    else []
                )

                num_views = max(
                    len(images), len(azimuths), len(elevations), len(distances)
                )
                if num_views == 0:
                    return (
                        None,
                        "At least one image or camera parameter must be provided.",
                    )

                total_components = []
                for i in range(num_views):
                    total_components.append(f"image_{i}")
                    total_components.append(f"camera_pose_{i}")

                denoise_mask_int = [
                    1 if comp in denoise_mask else 0 for comp in total_components
                ]

                if len(denoise_mask_int) != len(total_components):
                    return (
                        None,
                        f"Denoise mask length mismatch: expected {len(total_components)} components.",
                    )

                # Pad the input lists to num_views length
                images_padded = images + [] * (
                    num_views - len(images)
                )  # Do not add None
                azimuths_padded = azimuths + [None] * (num_views - len(azimuths))
                elevations_padded = elevations + [None] * (num_views - len(elevations))
                distances_padded = distances + [None] * (num_views - len(distances))

                print("=====================")
                print(azimuths_padded)
                print("=====================")
                print(elevations_padded)
                print("=====================")
                print(distances_padded)

                # Prepare values
                img2img_kwargs.update(
                    {
                        "image": images_padded,
                        "multiview_azimuths": azimuths_padded,
                        "multiview_elevations": elevations_padded,
                        "multiview_distances": distances_padded,
                        "multiview_focal_length": focal_length,  # Pass focal_length here
                        "is_multiview": True,
                        "denoise_mask": denoise_mask_int,
                        # 'predict_camera_poses': True,
                    }
                )
            else:
                img2img_kwargs.update({"image": images})
            output = self.pipe.img2img(**img2img_kwargs)

        output_images = output.images

        if task == "depth2image":
            processed_images = []
            for img in output.images:
                depth_map = np.array(
                    img.convert("L")
                )  # Convert to grayscale numpy array
                min_depth = depth_map.min()
                max_depth = depth_map.max()
                colorized = colorize_depth_maps(depth_map, min_depth, max_depth)[0]
                colorized = np.transpose(colorized, (1, 2, 0))
                colorized = (colorized * 255).astype(np.uint8)
                img_colorized = Image.fromarray(colorized)
                processed_images.append(img_colorized)
            output_images = processed_images + output.images
        elif task in ["boundingbox2image", "semanticmap2image"]:
            # Interpolate between input and output images
            processed_images = []
            for input_img, output_img in zip(images, output.images):
                input_img_resized = input_img.resize(output_img.size)
                blended_img = Image.blend(input_img_resized, output_img, alpha=0.5)
                processed_images.append(blended_img)
            output_images = processed_images + output.images

        output = []
        for i, out in enumerate(output_images):
            out_path = f"/tmp/out_{i}.png"
            out.save(out_path)
            output.append(Path(out_path))
        return output


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps with reversed colors.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # Normalize depth values to [0, 1]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    # Invert the depth values to reverse the colors
    depth = 1 - depth

    # Use the colormap
    cm = matplotlib.colormaps[cmap]
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # values from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored
