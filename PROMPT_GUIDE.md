# Prompt Guide

All examples are generated with a CFG of $4.2$, $50$ steps, and are non-cherrypicked unless otherwise stated. Negative prompt is set to:
```
monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation
```

## 1. Text-to-Image

### 1.1 Long and detailed prompts give (much) better results.

Since our training comprised of long and detailed prompts, the model is more likely to generate better images with detailed prompts.


The model shows good text adherence with long and complex prompts as in below images. We use the first $20$ prompts from [simoryu's examples](https://cloneofsimo.github.io/compare_aura_sd3/). For detailed prompts, results of other models, refer to the above link.

<p align="center">
  <img src="assets/promptguide_complex.jpg" alt="Text-to-Image results" width="800">
</p>


### 1.2 Resolution

The model generally works well with height and width in range of $[768; 1280]$ (height/width must be divisible by 16) for text-to-image. For other tasks, it performs best with resolution around $512$. 

## 2. ID Customization & Subject-driven generation

- The expected length of source captions is $30$ to $75$ words. Empirically, we find that longer prompt can help preserve the ID better but it might hinder the text-adherence for target caption.

- We find it better to add some descriptions (e.g., from source caption) to target to preserve the identity, especially for complex subjects with delicate details. 

<p align="center">
  <img src="assets/promptguide_idtask.jpg" alt="ablation id task" width="800">
</p>

## 3. Multiview generation

We recommend not use captions, which describe the facial features e.g., looking at the camera, etc, to mitigate multifaced/janus problems.

## 4. Image editing

We find it's generally better to set the guidance scale to lower value e.g., $[3; 3.5]$ to avoid over-saturation results.

## 5. Special tokens and available colors

### 5.1 Task Tokens

| Task                  | Token                      | Additional Tokens |
|:---------------------|:---------------------------|:------------------|
| Text to Image        | `[[text2image]]`           | |
| Deblurring           | `[[deblurring]]`           | |
| Inpainting           | `[[image_inpainting]]`     | |
| Canny-edge and Image       | `[[canny2image]]`          | |
| Depth and Image     | `[[depth2image]]`          | |
| Hed and Image      | `[[hed2img]]`              | |
| Pose and Image      | `[[pose2image]]`           | |
| Image editing with Instruction | `[[image_editing]]` | |
| Semantic map and Image| `[[semanticmap2image]]`    | `<#00FFFF cyan mask: object/to/segment>` |
| Boundingbox and Image     | `[[boundingbox2image]]`    | `<#00FFFF cyan boundingbox: object/to/detect>` |
| ID customization             | `[[faceid]]`               | `[[img0]] target/caption [[img1]] caption/of/source/image_1 [[img2]] caption/of/source/image_2 [[img3]] caption/of/source/image_3` |
| Multiview          | `[[multiview]]`            | |
| Subject-Driven      | `[[subject_driven]]`       | `<item: name/of/subject> [[img0]] target/caption/goes/here [[img1]] insert/source/caption` |


Note that you can replace the cyan color above with any from below table and have multiple additional tokens to detect/segment multiple classes.

### 5.2 Available colors


| Hex Code | Color Name |
|:---------|:-----------|
| #FF0000 | <span style="color: #FF0000">red</span> |
| #00FF00 | <span style="color: #00FF00">lime</span> |
| #0000FF | <span style="color: #0000FF">blue</span> |
| #FFFF00 | <span style="color: #FFFF00">yellow</span> |
| #FF00FF | <span style="color: #FF00FF">magenta</span> |
| #00FFFF | <span style="color: #00FFFF">cyan</span> |
| #FFA500 | <span style="color: #FFA500">orange</span> |
| #800080 | <span style="color: #800080">purple</span> |
| #A52A2A | <span style="color: #A52A2A">brown</span> |
| #008000 | <span style="color: #008000">green</span> |
| #FFC0CB | <span style="color: #FFC0CB">pink</span> |
| #008080 | <span style="color: #008080">teal</span> |
| #FF8C00 | <span style="color: #FF8C00">darkorange</span> |
| #8A2BE2 | <span style="color: #8A2BE2">blueviolet</span> |
| #006400 | <span style="color: #006400">darkgreen</span> |
| #FF4500 | <span style="color: #FF4500">orangered</span> |
| #000080 | <span style="color: #000080">navy</span> |
| #FFD700 | <span style="color: #FFD700">gold</span> |
| #40E0D0 | <span style="color: #40E0D0">turquoise</span> |
| #DA70D6 | <span style="color: #DA70D6">orchid</span> |
