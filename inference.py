import os
import torch
from onediffusion.diffusion.pipelines.onediffusion import OneDiffusionPipeline
from PIL import Image


device = torch.device('cuda:0')
pipeline = OneDiffusionPipeline.from_pretrained("lehduong/OneDiffusion").to(device=device, dtype=torch.bfloat16)
NEGATIVE_PROMPT = "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)



################################################################################################
## 1. Text-to-image
################################################################################################
output = pipeline(
    prompt="[[text2image]] A bipedal black cat wearing a huge oversized witch hat, a wizards robe, casting a spell,in an enchanted forest. The scene is filled with fireflies and moss on surrounding rocks and trees", 
    negative_prompt=NEGATIVE_PROMPT, 
    num_inference_steps=50,
    guidance_scale=4,
    height=1024, 
    width=1024,
)
output.images[0].save(f'{output_dir}/text2image_output.jpg')



################################################################################################
## 2. Semantic to Image (and any other condition2image tasks):
################################################################################################
images = [
    Image.open("assets/examples/semantic_map/dragon_birds_woman.webp")
]
prompt = "[[semanticmap2image]] <#00ffff Cyan mask: dragon> <#ff0000 yellow mask: bird> <#800080 purple mask: woman> A woman in a red dress with gold floral patterns stands in a traditional Japanese-style building. She has black hair and wears a gold choker and earrings. Behind her, a large orange and white dragon coils around the structure. Two white birds fly near her. The building features paper windows and a wooden roof with lanterns. The scene blends traditional Japanese architecture with fantastical elements, creating a mystical atmosphere."
# set the denoise mask to [1, 0] to denoise image from conditions                      
# by default, the height and width will be set so that input image is minimally cropped
ret = pipeline.img2img(
    image=images, 
    num_inference_steps=50, 
    prompt=prompt, 
    denoise_mask=[1, 0], 
    guidance_scale=4, 
    negative_prompt=NEGATIVE_PROMPT,
    # height=512,
    # width=512,
)
ret.images[0].save(f"{output_dir}/semanticmap2image_output.jpg")



################################################################################################
## 3. Depth Estimation (and any other image2condtition tasks):
################################################################################################
images = [
    Image.open("assets/examples/images/cat_on_table.webp"), 
]
prompt = "[[depth2image]] cat sitting on a table" # you can omit caption i.e., setting prompt to "[[depth2image]]"
# set the denoise mask to [0, 1] to denoise condition (depth, pose, hed, canny, semantic map etc)                        
# by default, the height and width will be set so that input image is minimally cropped
ret = pipeline.img2img(
    image=images, 
    num_inference_steps=50, 
    prompt=prompt, 
    denoise_mask=[0, 1], 
    guidance_scale=4, 
    NEGATIVE_PROMPT=NEGATIVE_PROMPT,
    # height=512,
    # width=512,
)
ret.images[0].save(f"{output_dir}/image2depth_output.jpg")



################################################################################################
## 4. ID Customization
################################################################################################
images = [
    Image.open("assets/examples/id_customization/chenhao/image_0.png"), 
    Image.open("assets/examples/id_customization/chenhao/image_1.png"), 
    Image.open("assets/examples/id_customization/chenhao/image_2.png")
]
##### we will set the denoise mask to [1, 0, 0, 0], which mean we the generated image will be the first view and next three views will be the condition (with same order as `images` list)
#### prompt format: [[faceid]] [[img0]] target/caption [[img1]] caption/of/first/image [[img2]] caption/of/second/image [[img3]] caption/of/third/image
prompt = "[[faceid]] \
    [[img0]] A woman dressed in traditional attire with intricate headpieces. She is looking at the camera and having neutral expression. \
    [[img1]] A woman with long dark hair, smiling warmly while wearing a floral dress. \
    [[img2]] A woman in traditional clothing holding a lace parasol, with her hair styled elegantly. \
    [[img3]] A woman in elaborate traditional attire and jewelry, with an ornate headdress, looking intently forward. \
"
# by default,  all images will be cropped according to the FIRST input image.
ret = pipeline.img2img(image=images, num_inference_steps=75, prompt=prompt, denoise_mask=[1, 0, 0, 0], guidance_scale=4, negative_prompt=NEGATIVE_PROMPT)
ret.images[0].save(f"{output_dir}/idcustomization_output.jpg")



################################################################################################
## 5. Image to multiview
################################################################################################
images = [
    Image.open("assets/examples/images/cat_on_table.webp"), 
]
prompt = "[[multiview]] A cat with orange and white fur sits on a round wooden table. The cat has striking green eyes and a pink nose. Its ears are perked up, and its tail is curled around its body. The background is blurred, showing a white wall, a wooden chair, and a wooden table with a white pot and green plant. A white curtain is visible on the right side. The cat's gaze is directed slightly to the right, and its paws are white. The overall scene creates a cozy, domestic atmosphere with the cat as the central focus."
# denoise mask: [0, 0, 1, 0, 1, 0, 1, 0] is for [img_0, camera of img0, img_1, camera of img1, img_2, camera of img2, img_3, camera of img3]
# since we provide 1 input image and all camera positions, the we don't need to denoise those value and set the mask to 0
# set the mask of [img_1, img_2, img_3] to 1 to generate novel views
# NOTE: only support SQUARE image
ret = pipeline.img2img(
    image=images, 
    num_inference_steps=60, 
    prompt=prompt, 
    negative_prompt=NEGATIVE_PROMPT,
    denoise_mask=[0, 0, 1, 0, 1, 0, 1, 0], 
    guidance_scale=4,
    multiview_azimuths=[0,20,40,60], # relative azimuth to first views
    multiview_elevations=[0,0,0,0], # relative elevation to first views
    multiview_distances=[1.5,1.5,1.5,1.5],
    # multiview_c2ws=None, # you can provide the camera-to-world matrix of shape [N, 4,4] for ALL views, camera extrinsics matrix is relative to first view
    # multiview_intrinsics=None, # provide the intrinsics matrix if c2ws is used
    height=512,
    width=512,
    is_multiview=True,
)
for i in range(len(ret.images)):
    ret.images[i].save(f"{output_dir}/img2multiview_output_view{i+1}.jpg")


