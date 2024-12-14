import os
import subprocess
import gradio as gr
import torch
import base64
import io
from PIL import Image
from transformers import (
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    T5EncoderModel, T5Tokenizer
)
from transformers import (
    AutoProcessor, AutoModelForCausalLM, GenerationConfig,
    T5EncoderModel, T5Tokenizer
)
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler, FluxPipeline
from onediffusion.diffusion.pipelines.onediffusion import OneDiffusionPipeline
from onediffusion.models.denoiser.nextdit import NextDiT
from onediffusion.dataset.utils import get_closest_ratio, ASPECT_RATIO_512
from typing import List, Optional
import matplotlib
import numpy as np
import cv2
import argparse
from PIL import Image

import csv
import ast
from gradio import components, utils
from typing import List, Any
from types import MethodType

os.environ["GRADIO_EXAMPLES_CACHE"] = "./assets/gradio_cached_examples"

# Task-specific tokens
TASK2SPECIAL_TOKENS = {
    "text2image": "[[text2image]]",
    "deblurring": "[[deblurring]]",
    "inpainting": "[[image_inpainting]]",
    "canny": "[[canny2image]]",
    "depth2image": "[[depth2image]]",
    "hed2image": "[[hed2img]]",
    "pose2image": "[[pose2image]]",
    "semanticmap2image": "[[semanticmap2image]]",
    "boundingbox2image": "[[boundingbox2image]]",
    "image_editing": "[[image_editing]]",
    "faceid": "[[faceid]]",
    "multiview": "[[multiview]]",
    "subject_driven": "[[subject_driven]]"
}
NEGATIVE_PROMPT = "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"


class LlavaCaptionProcessor:
    def __init__(self):
        model_name = "llava-hf/llama3-llava-next-8b-hf"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        self.SPECIAL_TOKENS = "assistant\n\n\n"

    def generate_response(self, image: Image.Image, msg: str) -> str:
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": msg}]}]
        with torch.no_grad():
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=200)
            response = self.processor.decode(output[0], skip_special_tokens=True)
        return response.split(msg)[-1].strip()[len(self.SPECIAL_TOKENS):]

    def process(self, images: List[Image.Image], msg: str = None) -> List[str]:
        if msg is None:
            msg = f"Describe the contents of the photo in 150 words or fewer."
        try:
            return [self.generate_response(img, msg) for img in images]
        except Exception as e:
            print(f"Error in process: {str(e)}")
            raise


class MolmoCaptionProcessor:
    def __init__(self):
        pretrained_model_name = 'cyan2k/molmo-7B-D-bnb-4bit'
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

    def generate_response(self, image: Image.Image, msg: str) -> str:
        inputs = self.processor.process(
            images=[image],
            text=msg
        )
        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        
        # Generate output
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=250, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )
        
        # Only get generated tokens and decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


    def process(self, images: List[Image.Image], msg: str = None) -> List[str]:
        if msg is None:
            msg = f"Describe the contents of the photo in 150 words or fewer."
        try:
            return [self.generate_response(img, msg) for img in images]
        except Exception as e:
            print(f"Error in process: {str(e)}")
            raise


class PlaceHolderCaptionProcessor:
    def __init__(self):
        pass

    def generate_response(self, image: Image.Image, msg: str) -> str:
        return ""
    
    def process(self, images: List[Image.Image], msg: str = None) -> List[str]:
        return [""] * len(images)
    
    
def initialize_models(captioner_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pipeline = OneDiffusionPipeline.from_pretrained("lehduong/OneDiffusion").to(device=device, dtype=torch.bfloat16)
    if captioner_name == 'molmo':
        captioner = MolmoCaptionProcessor()
    elif captioner_name == 'llava':
        captioner = LlavaCaptionProcessor()
    else:
        captioner = PlaceHolderCaptionProcessor()
    return pipeline, captioner

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


def format_prompt(task_type: str, captions: List[str]) -> str:
    if not captions:
        return ""
    if task_type == "faceid":
        img_prompts = [f"[[img{i}]] {caption}" for i, caption in enumerate(captions, start=1)]
        return f"[[faceid]] [[img0]] insert/your/caption/here {' '.join(img_prompts)}"
    elif task_type == "image_editing":
        return f"[[image_editing]] insert/your/instruction/here"
    elif task_type == "semanticmap2image":
        return f"[[semanticmap2image]] <#00ffff Cyan mask: insert/concept/to/segment/here> {captions[0]}"
    elif task_type == "boundingbox2image":
        return f"[[boundingbox2image]] <#00ffff Cyan boundingbox: insert/concept/to/segment/here> {captions[0]}"
    elif task_type == "multiview":
        img_prompts = captions[0]
        return f"[[multiview]] {img_prompts}"
    elif task_type == "subject_driven":
        return f"[[subject_driven]] <item: insert/item/here> [[img0]] insert/your/target/caption/here [[img1]] {captions[0]}"
    else:
        return f"{TASK2SPECIAL_TOKENS[task_type]} {captions[0]}"

def update_prompt(images: List[Image.Image], task_type: str, custom_msg: str = None):
    if not images:
        return format_prompt(task_type, []), "Please upload at least one image!"
    try:
        captions = captioner.process(images, custom_msg)
        if not captions:
            return "", "No valid images found!"
        prompt = format_prompt(task_type, captions)
        return prompt, f"Generated {len(captions)} captions successfully!"
    except Exception as e:
        return "", f"Error generating captions: {str(e)}"


def generate_image(images: List[Image.Image], prompt: str, negative_prompt: str, num_inference_steps: int, guidance_scale: float, 
                   denoise_mask: List[str], task_type: str, azimuth: str, elevation: str, distance: str, focal_length: float,
                   height: int = 1024, width: int = 1024, scale_factor: float = 1.0, scale_watershed: float = 1.0,
                   noise_scale: float = None, progress=gr.Progress()):
    try:
        img2img_kwargs = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'height': height,
            'width': width,
            'forward_kwargs': {
                'scale_factor': scale_factor,
                'scale_watershed': scale_watershed
            },
            'noise_scale': noise_scale  # Added noise_scale here
        }

        if task_type == 'multiview':
            # Parse azimuth, elevation, and distance into lists, allowing 'None' values
            azimuths = [float(a.strip()) if a.strip().lower() != 'none' else None for a in azimuth.split(',')] if azimuth else []
            elevations = [float(e.strip()) if e.strip().lower() != 'none' else None for e in elevation.split(',')] if elevation else []
            distances = [float(d.strip()) if d.strip().lower() != 'none' else None for d in distance.split(',')] if distance else []

            num_views = max(len(images), len(azimuths), len(elevations), len(distances))
            if num_views == 0:
                return None, "At least one image or camera parameter must be provided."

            total_components = []
            for i in range(num_views):
                total_components.append(f"image_{i}")
                total_components.append(f"camera_pose_{i}")

            denoise_mask_int = [1 if comp in denoise_mask else 0 for comp in total_components]

            if len(denoise_mask_int) != len(total_components):
                return None, f"Denoise mask length mismatch: expected {len(total_components)} components."

            # Pad the input lists to num_views length
            images_padded = images + [] * (num_views - len(images))  # Do not add None
            azimuths_padded = azimuths + [None] * (num_views - len(azimuths))
            elevations_padded = elevations + [None] * (num_views - len(elevations))
            distances_padded = distances + [None] * (num_views - len(distances))

            # Prepare values
            img2img_kwargs.update({
                'image': images_padded,
                'multiview_azimuths': azimuths_padded,
                'multiview_elevations': elevations_padded,
                'multiview_distances': distances_padded,
                'multiview_focal_length': focal_length,  # Pass focal_length here
                'is_multiview': True,
                'denoise_mask': denoise_mask_int,
                # 'predict_camera_poses': True,
            })
        else:
            total_components = ["image_0"] + [f"image_{i+1}" for i in range(len(images))]
            denoise_mask_int = [1 if comp in denoise_mask else 0 for comp in total_components]
            if len(denoise_mask_int) != len(total_components):
                return None, f"Denoise mask length mismatch: expected {len(total_components)} components."

            img2img_kwargs.update({
                'image': images,
                'denoise_mask': denoise_mask_int
            })

        progress(0, desc="Generating image...")
        if task_type == 'text2image':
            output = pipeline(
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height, 
                width=width,
                scale_factor=scale_factor,
                scale_watershed=scale_watershed,
                noise_scale=noise_scale  # Added noise_scale here
            )
        else:
            output = pipeline.img2img(**img2img_kwargs)
        progress(1, desc="Done!")

        # Process the output images if task is 'depth2image' and predicting depth
        if task_type == 'depth2image' and denoise_mask_int[-1] == 1:
            processed_images = []
            for img in output.images:
                depth_map = np.array(img.convert('L'))  # Convert to grayscale numpy array
                min_depth = depth_map.min()
                max_depth = depth_map.max()
                colorized = colorize_depth_maps(depth_map, min_depth, max_depth)[0]
                colorized = np.transpose(colorized, (1, 2, 0))
                colorized = (colorized * 255).astype(np.uint8)
                img_colorized = Image.fromarray(colorized)
                processed_images.append(img_colorized)
            output_images = processed_images + output.images
        elif task_type in ['boundingbox2image', 'semanticmap2image'] and denoise_mask_int == [0,1] and images:
            # Interpolate between input and output images
            processed_images = []
            for input_img, output_img in zip(images, output.images):
                input_img_resized = input_img.resize(output_img.size)
                blended_img = Image.blend(input_img_resized, output_img, alpha=0.5)
                processed_images.append(blended_img)
            output_images = processed_images + output.images
        else:
            output_images = output.images

        return output_images, "Generation completed successfully!"

    except Exception as e:
        return None, f"Error during generation: {str(e)}"

def update_denoise_checkboxes(images_state: List[Image.Image], task_type: str, azimuth: str, elevation: str, distance: str):
    if task_type == 'multiview':
        azimuths = [a.strip() for a in azimuth.split(',')] if azimuth else []
        elevations = [e.strip() for e in elevation.split(',')] if elevation else []
        distances = [d.strip() for d in distance.split(',')] if distance else []
        images_len = len(images_state)

        num_views = max(images_len, len(azimuths), len(elevations), len(distances))
        if num_views == 0:
            return gr.update(choices=[], value=[]), "Please provide at least one image or camera parameter."

        # Pad lists to the same length
        azimuths += ['None'] * (num_views - len(azimuths))
        elevations += ['None'] * (num_views - len(elevations))
        distances += ['None'] * (num_views - len(distances))
        # Do not add None to images_state

        labels = []
        values = []
        for i in range(num_views):
            labels.append(f"image_{i}")
            labels.append(f"camera_pose_{i}")

            # Default behavior: condition on provided inputs, generate missing ones
            if i >= images_len:
                values.append(f"image_{i}")
            if azimuths[i].lower() == 'none' or elevations[i].lower() == 'none' or distances[i].lower() == 'none':
                values.append(f"camera_pose_{i}")

        return gr.update(choices=labels, value=values)
    else:
        labels = ["image_0"] + [f"image_{i+1}" for i in range(len(images_state))]
        values = ["image_0"]
        return gr.update(choices=labels, value=values)

def apply_mask(images_state):
    if len(images_state) < 2:
        return None, "Please upload at least two images: first as the base image, second as the mask."
    base_img = images_state[0]
    mask_img = images_state[1]

    # Convert images to arrays
    base_arr = np.array(base_img)
    mask_arr = np.array(mask_img)

    # Convert mask to grayscale
    if mask_arr.ndim == 3:
        gray_mask = cv2.cvtColor(mask_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray_mask = mask_arr

    # Create a binary mask where non-black pixels are True
    binary_mask = gray_mask > 10

    # Define the gray color
    gray_color = np.array([128, 128, 128], dtype=np.uint8)

    # Apply gray color where mask is True
    masked_arr = base_arr.copy()
    masked_arr[binary_mask] = gray_color

    masked_img = Image.fromarray(masked_arr)
    return [masked_img], "Mask applied successfully!"

def process_images_for_task_type(images_state: List[Image.Image], task_type: str):
    # No changes needed here since we are processing the output images
    return images_state, images_state

def get_example():
    # Define example configurations and save images to temporary files
    examples = [
        [
            "Text to Image",  # Example name
            None,  # Preview column
            [],  # Empty list instead of None for input images
            "[[text2image]] A bipedal black cat wearing a huge oversized witch hat, a wizards robe, casting a spell,in an enchanted forest. The scene is filled with fireflies and moss on surrounding rocks and trees",
            NEGATIVE_PROMPT,
            50,  # num_steps
            4.0,  # guidance_scale
            ["image_0"],  # denoise_mask
            "text2image",  # task_type
            "0",  # azimuth
            "0",  # elevation
            "1.5",  # distance
            1.3887,  # focal_length
            1024,  # height
            1024,  # width
            1.0,  # scale_factor
            1.0,  # scale_watershed
            1.0,  # noise_scale
        ],
        
        [
            "ID Customization with 1 images",  # Example name - new column
            "./assets/examples/id_customization/chika/image_0.png",  # Preview first image
            [
                "./assets/examples/id_customization/chika/image_0.png",
            ],  # Input image paths
            "[[faceid]] [[img0]] photo depict a female anime character with pink hair and blue eyes,  sitting in a fine dining restaurant, black dress, smiling open mouth widely [[img1]] The photo depicts an anime-style cartoon character of a young woman with pink hair and blue eyes. She's wearing a black dress with white collar and cuffs, adorned with a red bow at the neckline and a red bow on the chest. A black bow tie decorates her hair. The character is standing in a classroom, with a green chalkboard featuring Asian characters visible behind her. The classroom has a white ceiling with brown trim and a window with a green curtain. The woman has a cheerful expression and appears to be in motion, as her hair is flowing. The overall scene is colorful and vibrant, capturing a moment of everyday life in an anime-inspired setting.",
            NEGATIVE_PROMPT,
            75,  # num_steps
            4.0,  # guidance_scale
            ["image_0"],  # denoise_mask
            "faceid",  # task_type
            "0",  # azimuth
            "0",  # elevation
            "1.5",  # distance
            1.3887,  # focal_length
            576,  # height
            448,  # width
            1.0,  # scale_factor
            1.0,  # scale_watershed
            1.0,  # noise_scale
        ],
        
        
        [
            "ID Customization with multiple input images",  # Example name - new column
            "./assets/examples/id_customization/chenhao/image_0.png",  # Preview first image
            [
                "./assets/examples/id_customization/chenhao/image_0.png",
                "./assets/examples/id_customization/chenhao/image_1.png",
                "./assets/examples/id_customization/chenhao/image_2.png",
            ],  # Input image paths
            "[[faceid]] [[img0]]  A woman with dark hair styled in an intricate updo, wearing a traditional orange and black outfit with elaborate gold embroidery. She has an elegant, poised expression, standing against a serene outdoor setting with classical architecture [[img1]] A young Asian woman with long dark hair and brown eyes smiles at the camera. She wears a red tank top with white flowers and green leaves. The background is blurred, with white and blue tones visible. The image has a slightly grainy quality. [[img2]] A young Asian woman in traditional attire stands against a brown background. She wears a white dress adorned with purple and green floral patterns. Her hair is styled in a bun, and she holds a small white lace umbrella with a gold handle. The image captures her elegant appearance and cultural dress. [[img3]] A woman in traditional Asian attire stands in front of a blurred building. She wears a green robe with floral designs and a black hat with lace. A man in a red robe and black hat stands behind her. The scene appears to be set in an Asian country.",
            NEGATIVE_PROMPT,
            50,  # num_steps
            4.0,  # guidance_scale
            ["image_0"],  # denoise_mask
            "faceid",  # task_type
            "0",  # azimuth
            "0",  # elevation
            "1.5",  # distance
            1.3887,  # focal_length
            608,  # height
            416,  # width
            1.0,  # scale_factor
            1.0,  # scale_watershed
            1.0,  # noise_scale
        ],
        
        [
            "Image to Multiview",  # Example name - new column
            "assets/examples/images/cat_on_table.webp",  # Preview column - no image for text-to-multiview
            ["assets/examples/images/cat_on_table.webp"],  # No input images
            "[[multiview]] A cat with orange and white fur sits on a round wooden table. The cat has striking green eyes and a pink nose. Its ears are perked up, and its tail is curled around its body. The background is blurred, showing a white wall, a wooden chair, and a wooden table with a white pot and green plant. A white curtain is visible on the right side. The cat's gaze is directed slightly to the right, and its paws are white. The overall scene creates a cozy, domestic atmosphere with the cat as the central focus.",
            NEGATIVE_PROMPT,
            60,  # num_steps 
            4.0,  # guidance_scale
            ["image_1", "image_2", "image_3"],  # denoise_mask
            "multiview",  # task_type
            "0,20,40,60",  # azimuth - four views
            "0,0,0,0",  # elevation - different angles
            "1.5,1.5,1.5,1.5",  # distance
            1.3887,  # focal_length
            512,  # height
            512,  # width
            1.0,  # scale_factor
            1.0,  # scale_watershed
            1.0,  # noise_scale
        ],
        
        [
            "Semantic to Image",  # Example name - new column
            "assets/examples/semantic_map/dragon_birds_woman.webp",  # Preview column
            ["assets/examples/semantic_map/dragon_birds_woman.webp"],  # Input image path
            "[[semanticmap2image]] <#00ffff Cyan mask: insert/concept/to/segment/here> A woman in a red dress with gold floral patterns stands in a traditional Japanese-style building. She has black hair and wears a gold choker and earrings. Behind her, a large orange and white dragon coils around the structure. Two white birds fly near her. The building features paper windows and a wooden roof with lanterns. The scene blends traditional Japanese architecture with fantastical elements, creating a mystical atmosphere.",
            NEGATIVE_PROMPT,
            50,  # num_steps
            4.0,  # guidance_scale
            ["image_0"],  # denoise_mask
            "semanticmap2image",  # task_type
            "0",  # azimuth
            "0",  # elevation
            "1.5",  # distance
            1.3887,  # focal_length
            672,  # height
            384,  # width
            1.0,  # scale_factor
            1.0,  # scale_watershed
            1.0,  # noise_scale
        ],
        
        [
            "Subject driven generation",  # Example name - new column
            "./assets/examples/subject_driven/chill_guy.jpg",  # Preview column
            ["./assets/examples/subject_driven/chill_guy.jpg"],  # Input image path
            "[[subject_driven]] <item: cartoon dog> [[img0]]  a cartoon character resembling a dog, sitting on a beach. The character has a long, narrow face with a black nose and brown eyes. It's wearing a gray sweatshirt, blue jeans rolled up at the bottom, and red sneakers with white soles. it has a slight smirk on its face [[img1]] The photo features a cartoon character resembling a do. The character has a long, narrow face with a black nose and brown eyes. It's wearing a gray sweatshirt, blue jeans rolled up at the bottom, and red sneakers with white soles. The character's hands are tucked into its pockets, and it has a slight smirk on its face. The background is a solid gray color, and the image has a hand-drawn, slightly blurry quality. The character's head is turned to the left, and its body is facing forward. The overall style is simple and cartoonish, with bold lines and limited shading.",
            NEGATIVE_PROMPT,
            70,  # num_steps
            4.0,  # guidance_scale
            ["image_0"],  # denoise_mask
            "subject_driven",  # task_type
            "0",  # azimuth
            "0",  # elevation
            "1.5",  # distance
            1.3887,  # focal_length
            512,  # height
            512,  # width
            1.0,  # scale_factor
            1.0,  # scale_watershed
            1.0,  # noise_scale
        ],
        
        [
            "Depth to Image",  # Example name - new column
            "./assets/examples/depth/astronaut.webp",  # Preview column
            ["./assets/examples/depth/astronaut.webp"],  # Input image path
            "[[depth2image]] The image depicts a futuristic astronaut standing on a rocky terrain with orange flowers. The astronaut is wearing a yellow suit with a helmet and is equipped with a backpack. The astronaut is looking up at a large, circular, glowing portal in the sky, which is surrounded by a halo of light. The portal is emitting a warm glow and is surrounded by a few butterflies. The sky is dark with stars, and there are distant mountains visible. The overall atmosphere of the image is one of exploration and wonder.",
            NEGATIVE_PROMPT,
            50,  # num_steps
            4.0,  # guidance_scale
            ["image_0"],  # denoise_mask
            "depth2image",  # task_type
            "0",  # azimuth
            "0",  # elevation
            "1.5",  # distance
            1.3887,  # focal_length
            608,  # height
            416,  # width
            1.0,  # scale_factor
            1.0,  # scale_watershed
            1.0,  # noise_scale
        ],
        
        [
            "Image to depth",  # Example name - new column
            "./assets/examples/images/cat.webp",  # Preview column
            ["./assets/examples/images/cat.webp"],  # Input image path
            "[[depth2image]] A kitten sits in a small boat on a rainy lake. The kitten wears a pink sweater and hat with a pom-pom. It has orange and white fur, and its paws are visible. The scene is misty and atmospheric, with trees and mountains in the background.",
            NEGATIVE_PROMPT,
            50,  # num_steps
            4.0,  # guidance_scale
            ["image_1"],  # denoise_mask
            "depth2image",  # task_type
            "0",  # azimuth
            "0",  # elevation
            "1.5",  # distance
            1.3887,  # focal_length
            512,  # height
            512,  # width
            1.0,  # scale_factor
            1.0,  # scale_watershed
            1.0,  # noise_scale
        ],
        
        [
            "Image Editing",  # Example name - new column
            "assets/examples/image_editing/astronaut.webp",  # Preview column
            ["assets/examples/image_editing/astronaut.webp"],  # Input image path
            "[[image_editing]] change it to winter and snowy weather",
            NEGATIVE_PROMPT,
            60,  # num_steps
            3.2,  # guidance_scale
            ["image_0"],  # denoise_mask
            "image_editing",  # task_type
            "0",  # azimuth
            "0",  # elevation
            "1.5",  # distance
            1.3887,  # focal_length
            512,  # height
            512,  # width
            1.0,  # scale_factor
            1.0,  # scale_watershed
            1.0,  # noise_scale
        ],
        
        [
            "Text to Multiview",  # Example name - new column
            None,  # Preview column - no image for text-to-multiview
            [],  # No input images
            "[[multiview]] The 3D scene features a striking black raven perched on a weathered rock in a rugged, mountainous landscape. Its glossy feathers shimmer with iridescent highlights, adding depth and realism. The background reveals a misty valley with rolling hills and a solitary stone cottage, exuding a sense of isolation and mystery. The earthy tones of the terrain, scattered with rocks and tufts of grass, contrast beautifully with the raven's dark plumage. The atmosphere feels serene yet haunting, evoking themes of solitude and nature's quiet power.",
            NEGATIVE_PROMPT,
            60,  # num_steps 
            4.0,  # guidance_scale
            ["image_0", "image_1", "image_2", "image_3"],  # denoise_mask
            "multiview",  # task_type
            "0,30,60,90",  # azimuth - four views
            "0,10,15,20",  # elevation - different angles
            "1.5,1.5,1.5,1.5",  # distance
            1.3887,  # focal_length
            512,  # height
            512,  # width
            1.0,  # scale_factor
            1.0,  # scale_watershed
            1.0,  # noise_scale
        ],
                
        [
            "Inpainting with Black mask",  # Example name - new column
            "./assets/examples/inpainting/giorno.webp",  # Preview column
            ["./assets/examples/inpainting/giorno.webp"],  # Input image path
            "[[image_inpainting]]",
            NEGATIVE_PROMPT,
            50,  # num_steps
            4.0,  # guidance_scale
            ["image_0"],  # denoise_mask
            "image_inpainting",  # task_type
            "0",  # azimuth
            "0",  # elevation
            "1.5",  # distance
            1.3887,  # focal_length
            416,  # height
            576,  # width
            1.0,  # scale_factor
            1.0,  # scale_watershed
            1.0,  # noise_scale
        ],
    ]
    return examples

def run_for_examples(example_name, preview_image, image_paths, prompt, negative_prompt, 
                    num_inference_steps, guidance_scale, denoise_mask, task_type, 
                    azimuth, elevation, distance, focal_length, height, width, 
                    scale_factor, scale_watershed, noise_scale):
    try:
        # Handle empty image paths or None
        images_state = []
        gallery_value = []
        
        if image_paths:  # Only process if image_paths is not None and not empty
            if isinstance(image_paths, list) and len(image_paths) > 0:
                for path in image_paths:
                    try:
                        if path is not None:
                            img = Image.open(path).convert('RGB')
                            images_state.append(img)
                            gallery_value.append(path)
                    except Exception as e:
                        print(f"Error loading image {path}: {str(e)}")
        
        # Generate output images
        output_images, status = generate_image(
            images_state, prompt, negative_prompt, num_inference_steps, guidance_scale,
            denoise_mask, task_type, azimuth, elevation, distance, focal_length,
            height, width, scale_factor, scale_watershed, noise_scale
        )
        
        # For preview gallery - show actual loaded images if any
        preview_images = images_state # if images_state else []
        
        return output_images, status, gallery_value, images_state, preview_images
    
    except Exception as e:
        return None, f"Error in example generation: {str(e)}", [], [], []

def update_gallery_state(files, current_state):
    """Update image state when new files are uploaded or cleared"""
    # Handle case when files is None or empty
    if not files:
        return [], [], []  # Return empty states for images, gallery, and preview
    
    # Ensure files is a list
    if not isinstance(files, list):
        files = [files]
        
    # Process new uploads
    processed_images = []
    for file in files:
        try:
            if isinstance(file, dict) and "name" in file:  # Handle file dict from gradio
                img_path = file["name"]
            elif isinstance(file, str):  # Handle direct file paths
                img_path = file
            elif isinstance(file, Image.Image):  # Handle PIL Image objects
                processed_images.append(file.convert('RGB'))
                continue
            else:
                print(f"Skipping unsupported file type: {type(file)}")
                continue
                
            img_pil = Image.open(img_path).convert('RGB')
            processed_images.append(img_pil)
            
        except Exception as e:
            return [], [], []
    
    # If no images were successfully processed, return empty states
    if not processed_images:
        return [], [], []
        
    # Return updated states and preview images
    # processed_images for the image state
    # files for the gallery state (original files)
    # processed_images again for preview
    return processed_images, files, processed_images

def delete_selected_images(selected_indices, images_state, gallery_state):
    """Delete selected images from gallery and state"""
    if not selected_indices or not images_state:
        return images_state, gallery_state, images_state, []
    
    # Create lists of indices to keep
    keep_indices = [i for i in range(len(images_state)) if i not in selected_indices]
    
    # Update image state
    updated_images = [images_state[i] for i in keep_indices]
    
    # Update gallery state if it exists
    updated_gallery = [gallery_state[i] for i in keep_indices] if gallery_state else []
    
    return updated_images, updated_gallery, updated_images, []

def delete_all_images():
    """Delete all images"""
    return [], [], [], []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the Gradio demo with specified captioner.')
    parser.add_argument('--captioner', type=str, choices=['molmo', 'llava', 'disable'], default='molmo', help='Captioner to use: molmo, llava, disable.')
    args = parser.parse_args()

    # Initialize models with the specified captioner
    pipeline, captioner = initialize_models(args.captioner)

    with gr.Blocks(title="OneDiffusion Demo") as demo:
        gr.Markdown("""
        # OneDiffusion Demo with (quantized) Molmo captioner

        **Welcome to the OneDiffusion Demo!**

        This application allows you to generate images based on your input prompts for various tasks. Here's how to use it:

        1. **Select Task Type**: Choose the type of task you want to perform from the "Task Type" dropdown menu.

        2. **Upload Images**: Drag and drop images directly onto the upload area, or click to select files from your device.

        3. **Generate Captions**: **If you upload any images**, Click the "Generate Captions" button to format the text prompt according to chosen task. In this demo, you will **NEED** to provide the caption of each source image manually. We recommend using Molmo for captioning.

        4. **Configure Generation Settings**: Expand the "Advanced Configuration" section to adjust parameters like the number of inference steps, guidance scale, image size, and more.

        5. **Generate Images**: After setting your preferences, click the "Generate Image" button. The generated images will appear in the "Generated Images" gallery.

        6. **Manage Images**: Use the "Delete Selected Images" or "Delete All Images" buttons to remove unwanted images from the gallery.

        **Notes**:
        - Check out the [Prompt Guide](https://github.com/lehduong/OneDiffusion/blob/main/PROMPT_GUIDE.md).
        
        - For text-to-image:
            + simply enter your prompt in this format "[[text2image]] your/prompt/here" and press the "Generate Image" button.
            
        - For boundingbox2image/semantic2image/inpainting etc tasks:
            + To perform condition-to-image such as semantic map to image, follow above steps
            + For image-to-condition e.g., image to depth, change the denoise_mask checkbox before generating images. You must UNCHECK image_0 box and CHECK image_1 box. Caption is not required for this task.
            
        - For FaceID tasks: 
            + Use 3 or 4 images if single input image does not give satisfactory results.
            + All images will be resized and center cropped to the input height and width. You should choose height and width so that faces in input images won't be cropped.
            + Model works best with close-up portrait (input and output) images.
            + If the model does not conform your text prompt, try using shorter caption for source image(s).
            + If you have non-human subjects and does not get satisfactory results, try "copying" part of caption of source images where it describes the properties of the subject e.g., a monster with red eyes, sharp teeth, etc.
            
        - For Multiview generation:
            + The input camera elevation/azimuth ALWAYS starts with 0. If you want to generate images of azimuths 30,60,90 and elevations of 10,20,30 (wrt input image), the correct input azimuth is: `0, 30, 60, 90`; input elevation is `0,10,20,30`. The camera distance will be `1.5,1.5,1.5,1.5`
            + Only support square images (ideally in 512x512 resolution).
            + Ensure the number of elevations, azimuths, and distances are equal. 
            + The model generally works well for 2-5 views (include both input and generated images). Since the model is trained with 3 views on 512x512 resolution, you might try scale_factor of [1.1; 1.5] and scale_watershed of [100; 400] for better extrapolation.
            + For better results:
                1) try increasing num_inference_steps to 75-100.
                2) avoid aggressively changes in target camera poses, for example to generate novel views at azimuth of 180, (simultaneously) generate 4 views with azimuth of 45, 90, 135, 180.
        
        Enjoy creating images with OneDiffusion!
        """)

        with gr.Row():
            with gr.Column():
                images_state = gr.State([])
                selected_indices_state = gr.State([])
                
                with gr.Row():
                    # Replace gallery with File input
                    gallery = gr.File(
                        label="Input Images",
                        file_count="multiple",
                        type="filepath",
                        file_types=["image"]
                    )
                
                # Add preview gallery
                preview_gallery = gr.Gallery(
                    label="Image Preview",
                    show_label=True,
                    columns=2,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )

                with gr.Row():
                    delete_button = gr.Button("Delete Selected Images")
                    delete_all_button = gr.Button("Delete All Images")
                
                task_type = gr.Dropdown(
                    choices=list(TASK2SPECIAL_TOKENS.keys()),
                    value="text2image",
                    label="Task Type"
                )
                
                captioning_message = gr.Textbox(
                    lines=2,
                    value="Describe the contents of the photo in 60 words.",
                    label="Custom message for captioner"
                )
                
                auto_caption_btn = gr.Button("Generate Captions")

            with gr.Column():
                prompt = gr.Textbox(
                    lines=3,
                    placeholder="Enter your prompt here or use auto-caption...",
                    label="Prompt"
                )
                negative_prompt = gr.Textbox(
                    lines=3,
                    value=NEGATIVE_PROMPT,
                    placeholder="Enter negative prompt here...",
                    label="Negative Prompt"
                )
                caption_status = gr.Textbox(label="Caption Status")
                
        num_steps = gr.Slider(
            minimum=1,
            maximum=200,
            value=50,
            step=1,
            label="Number of Inference Steps"
        )
        guidance_scale = gr.Slider(
            minimum=0.1,
            maximum=10.0,
            value=4,
            step=0.1,
            label="Guidance Scale"
        )
        height = gr.Number(value=1024, label="Height")
        width = gr.Number(value=1024, label="Width")
        
        with gr.Accordion("Advanced Configuration", open=False):
            with gr.Row():
                denoise_mask_checkbox = gr.CheckboxGroup(
                    label="Denoise Mask",
                    choices=["image_0"],
                    value=["image_0"]
                )
                azimuth = gr.Textbox(
                    value="0",
                    label="Azimuths (degrees, comma-separated, 'None' for missing)"
                )
                elevation = gr.Textbox(
                    value="0",
                    label="Elevations (degrees, comma-separated, 'None' for missing)"
                )
                distance = gr.Textbox(
                    value="1.5",
                    label="Distances (comma-separated, 'None' for missing)"
                )
                focal_length = gr.Number(
                    value=1.3887,
                    label="Focal Length of camera for multiview generation"
                )
                scale_factor = gr.Number(value=1.0, label="Scale Factor")
                scale_watershed = gr.Number(value=1.0, label="Scale Watershed")
                noise_scale = gr.Number(value=1.0, label="Noise Scale")  # Added noise_scale input

        output_images = gr.Gallery(
            label="Generated Images",
            show_label=True,
            columns=4,
            rows=2,
            height="auto",
            object_fit="contain"
        )
        
        with gr.Column():
            generate_btn = gr.Button("Generate Image")
            # apply_mask_btn = gr.Button("Apply Mask")
        
        status = gr.Textbox(label="Generation Status")

        def update_height_width(images_state):
            if images_state:
                closest_ar = get_closest_ratio(
                    height=images_state[0].size[1],
                    width=images_state[0].size[0],
                    ratios=ASPECT_RATIO_512
                )
                height_val, width_val = int(closest_ar[0][0]), int(closest_ar[0][1])
            else:
                height_val, width_val = 1024, 1024  # Default values
            return gr.update(value=height_val), gr.update(value=width_val)
        
        # Update the image selection handler
        def on_select(evt: gr.SelectData, selected_indices):
            """Handle image selection in gallery"""
            if selected_indices is None:
                selected_indices = []
            
            if evt.index in selected_indices:
                selected_indices.remove(evt.index)
            else:
                selected_indices.append(evt.index)
            return selected_indices

        # Connect gallery upload
        gallery.upload(
            fn=update_gallery_state,
            inputs=[gallery, images_state],
            outputs=[images_state, gallery, preview_gallery],
            show_progress="full"
        ).then(
            fn=update_height_width,
            inputs=[images_state],
            outputs=[height, width]
        ).then(
            fn=update_denoise_checkboxes,
            inputs=[images_state, task_type, azimuth, elevation, distance],
            outputs=[denoise_mask_checkbox]
        )

        # Update delete buttons connections
        delete_button.click(
            fn=delete_selected_images,
            inputs=[selected_indices_state, images_state, gallery],
            outputs=[images_state, gallery, preview_gallery, selected_indices_state]
        ).then(
            fn=update_height_width,
            inputs=[images_state],
            outputs=[height, width]
        ).then(
            fn=update_denoise_checkboxes,
            inputs=[images_state, task_type, azimuth, elevation, distance],
            outputs=[denoise_mask_checkbox]
        )

        delete_all_button.click(
            fn=delete_all_images,
            inputs=[],
            outputs=[images_state, gallery, preview_gallery, selected_indices_state]
        ).then(
            fn=update_denoise_checkboxes,
            inputs=[images_state, task_type, azimuth, elevation, distance],
            outputs=[denoise_mask_checkbox]
        ).then(
            fn=update_height_width,
            inputs=[images_state],
            outputs=[height, width]
        )


        task_type.change(
            fn=update_denoise_checkboxes,
            inputs=[images_state, task_type, azimuth, elevation, distance],
            outputs=[denoise_mask_checkbox]
        )

        azimuth.change(
            fn=update_denoise_checkboxes,
            inputs=[images_state, task_type, azimuth, elevation, distance],
            outputs=[denoise_mask_checkbox]
        )

        elevation.change(
            fn=update_denoise_checkboxes,
            inputs=[images_state, task_type, azimuth, elevation, distance],
            outputs=[denoise_mask_checkbox]
        )

        distance.change(
            fn=update_denoise_checkboxes,
            inputs=[images_state, task_type, azimuth, elevation, distance],
            outputs=[denoise_mask_checkbox]
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                images_state, prompt, negative_prompt, num_steps, guidance_scale,
                denoise_mask_checkbox, task_type, azimuth, elevation, distance,
                focal_length, height, width, scale_factor, scale_watershed, noise_scale  # Added noise_scale here
            ],
            outputs=[output_images, status],
            concurrency_id="gpu_queue"
        )

        auto_caption_btn.click(
            fn=update_prompt,
            inputs=[images_state, task_type, captioning_message],
            outputs=[prompt, caption_status],
            concurrency_id="gpu_queue"
        )
        
        # apply_mask_btn.click(
        #     fn=apply_mask,
        #     inputs=[images_state],
        #     outputs=[output_images, status]
        # )
        
        # Update the Examples component with preview column
        examples = gr.Examples(
            examples=get_example(),
            fn=run_for_examples,
            inputs=[
                gr.Textbox(visible=False),  # Example name column
                gr.Image(show_label=False, visible=False),  # Preview column
                gallery,
                prompt,
                negative_prompt,
                num_steps,
                guidance_scale,
                denoise_mask_checkbox,
                task_type,
                azimuth,
                elevation,
                distance,
                focal_length,
                height,
                width,
                scale_factor,
                scale_watershed,
                noise_scale
            ],
            outputs=[
                output_images, 
                status, 
                gallery,
                images_state,
                preview_gallery
            ],
            cache_examples=True,
            label="Examples"
        )
        
        # the default load_from_cache function throws error for text-to-image and text-to-multiview because the list of input views is empty 
        def custom_load_from_cache(self, example_id: int) -> List[Any]:
            """Loads a particular cached example for the interface.
            Parameters:
                example_id: The id of the example to process (zero-indexed).
            """
            with open(self.cached_file, encoding="utf-8") as cache:
                examples = list(csv.reader(cache))
            example = examples[example_id + 1]  # +1 to adjust for header
            output = []
            if self.outputs is None:
                raise ValueError("self.outputs is missing")
            for component, value in zip(self.outputs, example):
                value_to_use = value
                try:
                    value_as_dict = ast.literal_eval(value)
                    # File components that output multiple files get saved as a python list
                    # need to pass the parsed list to serialize
                    # TODO: Better file serialization in 4.0
                    if isinstance(value_as_dict, list) and isinstance(
                        component, components.File
                    ):
                        tmp = value_as_dict
                    if not utils.is_prop_update(tmp):
                        raise TypeError("value wasn't an update")  # caught below
                    value_to_use = tmp
                    output.append(value_to_use)
                except (ValueError, TypeError, SyntaxError):
                    output.append(component.read_from_flag(value_to_use))
            return output
                        
        def apply_custom_load_from_cache(examples_instance):
            """
            Applies the custom load_from_cache method to a Gradio Examples instance.
            
            Parameters:
                examples_instance: The Gradio Examples instance to modify
            """
            examples_instance.load_from_cache = MethodType(custom_load_from_cache, examples_instance)
            
        apply_custom_load_from_cache(examples)

        # Connect the event handler for file upload changes
        gallery.change(
            fn=update_gallery_state,
            inputs=[gallery, images_state],
            outputs=[images_state, gallery, preview_gallery],
            show_progress="full"
        ).then(
            fn=update_height_width,
            inputs=[images_state],
            outputs=[height, width]
        ).then(
            fn=update_denoise_checkboxes,
            inputs=[images_state, task_type, azimuth, elevation, distance],
            outputs=[denoise_mask_checkbox]
        )

    demo.launch()