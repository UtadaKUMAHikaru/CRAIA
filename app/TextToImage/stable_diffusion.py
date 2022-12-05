

# import torch
from diffusers import StableDiffusionPipeline


device="cpu"

# pipe = StableDiffusionPipeline.from_pretrained("/Users/chendeen/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4")
# pipe = StableDiffusionPipeline.from_pretrained("./app/TextToImage/models--CompVis--stable-diffusion-v1-4", use_auth_token=True)
# pipe = StableDiffusionPipeline.from_pretrained("./app/TextToImage/models--CompVis--stable-diffusion-v1-4", use_auth_token=True)
# pipe = StableDiffusionPipeline.from_pretrained("app/TextToImage/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096")
# pipe = StableDiffusionPipeline.from_pretrained("app/TextToImage/models--CompVis--stable-diffusion-v1-4/")
# pipe = StableDiffusionPipeline.from_pretrained("app/TextToImage/models--CompVis--stable-diffusion-v1-4/")
# pipe = StableDiffusionPipeline.from_pretrained("app/TextToImage/models--CompVis--stable-diffusion-v1-4/", use_auth_token=TOKEN)
# pipe = StableDiffusionPipeline.from_pretrained("app/TextToImage/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096", use_auth_token=TOKEN)
# pipe = StableDiffusionPipeline.from_pretrained("/Users/chendeen/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096")  
# pipe = StableDiffusionPipeline.from_pretrained("/Users/chendeen/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096")  
# pipe = StableDiffusionPipeline.from_pretrained("/Users/chendeen/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096", revision="fp16", torch_dtype=torch.float16)  
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)  
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")  
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=TOKEN)  

# pipe = StableDiffusionPipeline.from_pretrained("/Users/chendeen/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/114c79c111c5836a89768787ba5b4ceee25bb734")  
pipe = StableDiffusionPipeline.from_pretrained("app/TextToImage/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096")  
# pipe = StableDiffusionPipeline.from_pretrained("/Users/chendeen/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/a304b1ab1b59dd6c3ba9c40705c29c6de4144096")  


pipe.to(device)
# pipe = pipe.to("cuda")

def text_to_image_simple(text = "a photograph of an astronaut riding a horse",init_img=None, parameters=None):

    prompt=text
    image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

    return image

    # Now to display an image you can do either save it such as:
    # image.save(f"astronaut_rides_horse.png")

# or if you're in a google colab you can directly display it with 
# image

# return_cached_folder


# /Users/chendeen/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4


