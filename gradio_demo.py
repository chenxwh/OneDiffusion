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
            output = self.model.generate(**inputs, max_new_tokens=250)
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

with gr.Blocks(title="OneDiffusion Demo") as demo:
    gr.Markdown("""
    # OneDiffusion Demo

    **Welcome to the OneDiffusion Demo!**

    This application allows you to generate images based on your input prompts for various tasks. Here's how to use it:

    1. **Select Task Type**: Choose the type of task you want to perform from the "Task Type" dropdown menu.

    2. **Upload Images**: Drag and drop images directly onto the upload area, or click to select files from your device.

    3. **Generate Captions**: **If you upload any images**, Click the "Generate Captions" button to generate descriptive captions for your uploaded images (depend on the task). You can enter a custom message in the "Custom Message for captioner" textbox e.g., "caption in 30 words" instead of 50 words.

    4. **Configure Generation Settings**: Expand the "Advanced Configuration" section to adjust parameters like the number of inference steps, guidance scale, image size, and more.

    5. **Generate Images**: After setting your preferences, click the "Generate Image" button. The generated images will appear in the "Generated Images" gallery.

    6. **Manage Images**: Use the "Delete Selected Images" or "Delete All Images" buttons to remove unwanted images from the gallery.

    **Notes**:
    - Check out the [Prompt Guide](https://github.com/lehduong/OneDiffusion/blob/main/PROMPT_GUIDE.md).
    
    - For text-to-image:
        + simply enter your prompt in this format "[[text2image]] your/prompt/here" and press the "Generate Image" button.
        
    - For boundingbox2image/semantic2image/inpainting etc tasks:
        + To perform condition-to-image such as semantic map to image, follow above steps
        + For image-to-condition e.g., image to depth, change the denoise_mask checkbox before generating images. You must UNCHECK image_0 box and CHECK image_1 box.
        
    - For FaceID tasks: 
        + Use 3 or 4 images if single input image does not give satisfactory results.
        + All images will be resized and center cropped to the input height and width. You should choose height and width so that faces in input images won't be cropped.
        + Model works best with close-up portrait (input and output) images.
        + If the model does not conform your text prompt, try using shorter caption for source image(s).
        + If you have non-human subjects and does not get satisfactory results, try "copying" part of caption of source images where it describes the properties of the subject e.g., a monster with red eyes, sharp teeth, etc.
        
    - For Multiview generation:
        + The input camera elevation/azimuth ALWAYS starts with $0$. If you want to generate images of azimuths 30,60,90 and elevations of 10,20,30 (wrt input image), the correct input azimuth is: `0, 30, 60, 90`; input elevation is `0,10,20,30`. The camera distance will be `1.5,1.5,1.5,1.5`
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
                gallery = gr.Gallery(
                    label="Input Images",
                    show_label=True,
                    columns=2,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )
            
            # In the UI section, update the file_output component:
            file_output = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Drag and drop images here or click to upload",
                height=100,
                scale=2,
                type="filepath"  # Add this parameter
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

    # Event Handlers
    def update_gallery(files, images_state):
        if not files:
            return images_state, images_state
        
        new_images = []
        for file in files:
            try:
                # Handle both file paths and file objects
                if isinstance(file, dict):  # For drag and drop files
                    file = file['path']
                elif hasattr(file, 'name'):  # For uploaded files
                    file = file.name
                    
                img = Image.open(file).convert('RGB')
                new_images.append(img)
            except Exception as e:
                print(f"Error loading image: {str(e)}")
                continue
                
        images_state.extend(new_images)
        return images_state, images_state

    def on_image_select(evt: gr.SelectData, selected_indices_state):
        selected_indices = selected_indices_state or []
        index = evt.index
        if index in selected_indices:
            selected_indices.remove(index)
        else:
            selected_indices.append(index)
        return selected_indices

    def delete_images(selected_indices, images_state):
        updated_images = [img for i, img in enumerate(images_state) if i not in selected_indices]
        selected_indices_state = []
        return updated_images, updated_images, selected_indices_state

    def delete_all_images(images_state):
        updated_images = []
        selected_indices_state = []
        return updated_images, updated_images, selected_indices_state

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

    # Connect events
    file_output.change(
        fn=update_gallery,
        inputs=[file_output, images_state],
        outputs=[images_state, gallery]
    ).then(
        fn=update_height_width,
        inputs=[images_state],
        outputs=[height, width]
    ).then(
        fn=update_denoise_checkboxes,
        inputs=[images_state, task_type, azimuth, elevation, distance],
        outputs=[denoise_mask_checkbox]
    )

    gallery.select(
        fn=on_image_select,
        inputs=[selected_indices_state],
        outputs=[selected_indices_state]
    )

    delete_button.click(
        fn=delete_images,
        inputs=[selected_indices_state, images_state],
        outputs=[images_state, gallery, selected_indices_state]
    ).then(
        fn=update_denoise_checkboxes,
        inputs=[images_state, task_type, azimuth, elevation, distance],
        outputs=[denoise_mask_checkbox]
    )

    delete_all_button.click(
        fn=delete_all_images,
        inputs=[images_state],
        outputs=[images_state, gallery, selected_indices_state]
    ).then(
        fn=update_denoise_checkboxes,
        inputs=[images_state, task_type, azimuth, elevation, distance],
        outputs=[denoise_mask_checkbox]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the Gradio demo with specified captioner.')
    parser.add_argument('--captioner', type=str, choices=['molmo', 'llava', 'disable'], default='molmo', help='Captioner to use: molmo, llava, disable.')
    args = parser.parse_args()

    # Initialize models with the specified captioner
    pipeline, captioner = initialize_models(args.captioner)

    demo.launch(debug=True)