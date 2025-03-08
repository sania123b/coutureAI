!pip install diffusers transformers accelerate pillow torch torchvision torchaudio
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display

# Load Stable Diffusion Model (Ensuring Structure & Quality)
try:
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    pipe.to("cuda")  
except Exception as e:
    print(f"Error loading model: {e}")

# Get User Inputs
gender = input("Enter gender (girl/boy): ").strip().lower()
clothing_type = input("Enter clothing type (saree, half saree, long frock, short frock, kurti, lehenga, etc.): ").strip().lower()
material = input("Enter material (cotton, silk, net, georgette, etc.): ").strip().lower()
sleeve_type = input("Enter sleeve type (full sleeves, sleeveless, puff sleeves, ruffle sleeves, etc.): ").strip().lower()
neck_type = input("Enter neck type (boat neck, round neck, sweetheart neck, V-shaped neck, etc.): ").strip().lower()
occasion = input("Enter occasion (wedding, festival, party, casual, etc.): ").strip().lower()
background = input("Enter background/place (temple, palace, garden, studio, etc.): ").strip().lower()
extra_features = input("Enter any extra features (golden embroidery, mirror work, zari border, etc.): ").strip().lower()
color = input("Enter primary color of the outfit: ").strip().lower()

# Enforcing Detailed Prompting
prompt = (f"A {gender} wearing a beautiful {color} {material} {clothing_type} with {sleeve_type} and a {neck_type}. "
          f"The outfit is designed for {occasion} and is set in a {background}. "
          f"It features {extra_features}. The hands and neck exactly match the description, ensuring high accuracy. "
          f"The outfit has high detail, intricate fabric textures, elegant draping, and realistic skin. "
          f"The image is in ultra-HD, with smooth skin, symmetrical face, and perfect hand positioning. "
          f"Soft lighting, cinematic quality, highly detailed 8K rendering.")

# Negative Prompt (To Remove Random Distortions)
negative_prompt = ("distorted face, extra fingers, incorrect clothing folds, blurry textures, "
                   "wrong sleeve type, incorrect neck type, missing details, low quality, unrealistic features")

# Generate Image
print("\n⏳ Generating image... Please wait...")
image = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=9).images[0]

# Display Image in Colab
display(image)

# Save Locally
image.save("generated_image.png")
print("\n✅ Image saved as 'generated_image.png'. You can download it from Colab Files.")
