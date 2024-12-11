import torch
from onediffusion.diffusion.pipelines.onediffusion import OneDiffusionPipeline
from PIL import Image

device = torch.device('cuda:0')
pipeline = OneDiffusionPipeline.from_pretrained("lehduong/OneDiffusion").to(device=device, dtype=torch.bfloat16)

NEGATIVE_PROMPT = "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"

## Text-to-image
output = pipeline(
    prompt="A bipedal black cat wearing a huge oversized witch hat, a wizards robe, casting a spell,in an enchanted forest. The scene is filled with fireflies and moss on surrounding rocks and trees", 
    negative_prompt=NEGATIVE_PROMPT, 
    num_inference_steps=50,
    guidance_scale=4,
    height=1024, 
    width=1024,
)
output.images[0].save('text2image_output.jpg')

## ID Customization
image = [
    Image.open("assets/examples/id_customization/image_0.png"), 
    Image.open("assets/examples/id_customization/image_1.png"), 
    Image.open("assets/examples/id_customization/image_2.png")
]

# input = [noise, cond_1, cond_2, cond_3]
prompt = "[[faceid]] \
    [[img0]] A woman dressed in traditional attire with intricate headpieces, posing gracefully with a serene expression. \
    [[img1]] A woman with long dark hair, smiling warmly while wearing a floral dress. \
    [[img2]] A woman in traditional clothing holding a lace parasol, with her hair styled elegantly. \
    [[img3]] A woman in elaborate traditional attire and jewelry, with an ornate headdress, looking intently forward. \
"
                        
ret = pipeline.img2img(image=image, num_inference_steps=75, prompt=prompt, denoise_mask=[1, 0, 0, 0], guidance_scale=4)
ret.images[0].save("idcustomization_output.jpg")
