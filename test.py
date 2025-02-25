import os
from ruamel.yaml import YAML
import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from model import model_types
from config import parse_args
from utils.utils_model import load_model
from utils.utils_data import get_test_data
from PIL import Image

def show_images(images):
    images = [np.array(image) for image in images]
    images = np.concatenate(images, axis=1)
    return Image.fromarray(images)

def main():
    args = parse_args()
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'test_config.yaml'), 'w'))
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet")
    if args.scheduler == 'ddim': scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,num_train_timesteps=1000,steps_offset=1)
    elif args.scheduler == 'pndm': scheduler = PNDMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    elif args.scheduler == 'ddpm': scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    fixed_vectors = torch.load("optimized_vectors/luxury_car.pt")
    mlp=model_types["ModifiedMLP"](resolution=args.resolution//64, fixed_vectors = fixed_vectors)
    unet.set_controlnet(mlp)
    load_model(unet, args.output_dir+'/unet.pth')
    model=StableDiffusionPipeline(vae=vae,text_encoder=text_encoder,tokenizer=tokenizer,unet=unet,scheduler=scheduler,safety_checker=None,feature_extractor=None,requires_safety_checker=False,)
    model=model.to(args.device)
    dataloader=get_test_data(data_dir=args.train_data_dir, given_prompt=args.prompt, given_concept=args.concept, max_concept_length=100)
    evaluate(model=model, dataloader=dataloader, args=args)

def predict_cond(model, prompt, seed, condition, device, img_size,num_inference_steps=50,interpolator=None, negative_prompt=None):
    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
    output = model(prompt=prompt, height=img_size, width=img_size, num_inference_steps=num_inference_steps, generator=generator, controlnet_cond=condition,controlnet_interpolator=interpolator,negative_prompt=negative_prompt)
    image = output[0][0]
    return image

def evaluate(model, dataloader, args):
    save_image_dir = os.path.join(args.output_dir, args.image_dir)
    save_vector_dir = os.path.join(args.output_dir, args.vector_dir)
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_vector_dir, exist_ok=True)
    images, seed = [], 0
    for i in range(args.num_test_samples):
        images = []
        for prompt,concept in zip(*dataloader):
            image = predict_cond(model=model,prompt=prompt,seed=seed,condition=concept,device=args.device,img_size=args.resolution,num_inference_steps=args.num_inference_steps,negative_prompt=args.negative_prompt)
            images.append(image)
        seed += 1
        images = show_images(images)
        images.save(f"{save_image_dir}/eval{i}.jpg")

if __name__ == "__main__":
    main()