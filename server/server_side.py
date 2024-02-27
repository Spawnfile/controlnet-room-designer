import io
import gc
import base64
import http

import numpy as np
import torch
from PIL import Image
from controlnet_aux import MLSDdetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Endpoint data format
class Item(BaseModel):
    image_data: str
    prompt: str 
    negative_prompt: str
    num_samples: int = 1 
    guidance_scale: int = 9
    inference_steps: int = 20
    seed: int = 12345

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_mlsd", torch_dtype=torch.float16)
preprocessor = MLSDdetector.from_pretrained("lllyasviel/Annotators")
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload() # Models will only be taken to the GPU when they are used, otherwise they will sit in the CPU for providing memory reducement.
pipe.enable_xformers_memory_efficient_attention() # using fast attention for fast inference

pipe.to('cuda')

def clean():
    torch.cuda.empty_cache()
    gc.collect()

def prompt_prep(prompt, negative_prompt, num_samples, seed):
    ret_prompt = [prompt for _ in range(num_samples)]
    ret_neg_prompt = [negative_prompt for _ in range(num_samples)]
    if num_samples == 1:
        generator = [torch.Generator(device="cuda").manual_seed(seed)]
    else:
        generator = [torch.Generator(device="cuda").manual_seed(seed), torch.Generator(device="cuda").manual_seed(np.random.randint(0, 2147483647))]
    return ret_prompt, ret_neg_prompt, generator

def predict(image, 
            prompt="a Japanese style living room, best quality, extremely detailed", 
            num_samples=1, 
            guidance_scale=9, 
            inference_steps=20, 
            negative_prompt="fewer digits, cropped, worst quality, low quality", 
            seed=12345):
    image = np.array(image)

    processed_image_mlsd = preprocessor(image, detect_resolution=512, image_resolution=768)
    clean()

    prompt, negative_prompt, generator = prompt_prep(prompt, negative_prompt, num_samples, seed)

    output = pipe(prompt=prompt,
                  image=processed_image_mlsd,
                  negative_prompt=negative_prompt,
                  num_inference_steps=inference_steps,
                  guidance_scale=guidance_scale,
                  generator=generator,)
    clean()

    return output.images

@app.post("/inference")
async def inference(item_raw: dict):
    try:
        item = Item(**item_raw['item'])
        prompt=item.prompt
        negative_prompt=item.negative_prompt
        num_samples=item.num_samples
        guidance_scale=item.guidance_scale
        inference_steps=item.inference_steps
        seed=item.seed

        img_data = base64.b64decode(item.image_data.encode('utf8'))
        img_stream = io.BytesIO(img_data)
        img = Image.open(img_stream)  

    except ValueError as e:
        raise http.HTTPException(status_code=422, detail=f"Invalid request format, {e}")

    output_list = predict(
                            image=img,
                            prompt=prompt, 
                            num_samples=num_samples, 
                            guidance_scale=guidance_scale, 
                            inference_steps=inference_steps, 
                            negative_prompt=negative_prompt, 
                            seed=seed)

    parsed_output = []
    for _, img in enumerate(output_list):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_byte = buffered.getvalue()
        img_b64 = base64.b64encode(img_byte).decode("utf-8")
        parsed_output.append(img_b64)

    return {"message": "Inference successfull", "output":parsed_output}
