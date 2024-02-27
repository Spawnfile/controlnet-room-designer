import io
import requests
import json
import base64

import gradio as gr
import numpy as np
from PIL import Image
from pydantic import BaseModel

class Item(BaseModel):
    image_data: str
    prompt: str 
    negative_prompt: str
    num_samples: int = 1 
    guidance_scale: int = 9
    inference_steps: int = 20
    seed: int = 12345

def encode_image(filename=None, img_arr=None):
    if filename is not None:
        with Image.open(filename) as img:
            buf = io.BytesIO()
            img.convert('RGB').save(buf, format="JPEG")
            img_np_array = np.ascontiguousarray(np.array(img))
    elif img_arr is not None: 
        img_np_array = np.ascontiguousarray(img_arr)
    else:
        raise ValueError("Either filename or img_arr needs to be present.")

    if len(img_np_array.shape) != 3 or img_np_array.shape[-1] != 3:
        raise ValueError("Input must be a 3D NumPy array with dimensions H x W x 3.")

    jpeg_bytes = io.BytesIO()
    Image.fromarray(img_np_array).save(jpeg_bytes, format="JPEG")
    image_data = base64.b64encode(jpeg_bytes.getvalue()).decode()
    return image_data

def parse_response(response: requests.models.Response):
    raw_outputs = json.loads(response.text)["output"]
    decoded_images = [base64.b64decode(image_data) for image_data in raw_outputs]
    pil_images = [Image.open(io.BytesIO(decoded_image)) for decoded_image in decoded_images]
    return pil_images

def send_request(url, request):
    headers = {'Content-type': 'application/json'}
    payload = json.dumps({'item': request.__dict__}, indent=2)
    response = requests.post(url, data=payload, headers=headers)
    print(f"Response status code: {response.status_code}")
    return response

def predict_request(input_image, prompt, num_samples, guidance_scale, inference_steps, negative_prompt, seed):

    image_data = encode_image(filename=None, img_arr=input_image)

    url = "http://127.0.0.1:80/inference"
    request = Item(\
                image_data=image_data,
                prompt=prompt, 
                negative_prompt=negative_prompt,
                num_samples=num_samples,
                guidance_scale= guidance_scale,
                inference_steps= inference_steps,
                seed=seed
                )

    response = send_request(url, request)
    image_list = parse_response (response)
    return image_list


block = gr.Blocks(theme=gr.themes.Monochrome()).queue()
with block:
    with gr.Row():
        gr.Markdown("## Interior Design")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt", value="a Japanese style living room, best quality, extremely detailed", placeholder="a Japanese style living room, best quality, extremely detailed")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="fewer digits, cropped, worst quality, low quality", 
                                placeholder="fewer digits, cropped, worst quality, low quality")
            run_button = gr.Button(value="Predict")
            num_samples = gr.Slider(label="Output Images", minimum=1, maximum=2, value=1, step=1)
            inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=200, value=20, step=1)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=10, value=9, step=0.1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", columns=num_samples, object_fit='fill')

    inputs = [input_image, prompt, num_samples, guidance_scale, inference_steps, negative_prompt, seed]
    run_button.click(fn=predict_request, inputs=inputs, outputs=result_gallery)


if __name__ == '__main__':
    print("Program Started.")
    block.launch(server_name='0.0.0.0', debug=True, share=True)
