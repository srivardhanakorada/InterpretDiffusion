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
from utils.utils_data import get_test_data, get_i2p_data
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
from metrics.nudenet_classify import detect_nude_and_q16
from metrics.CLIP_classify import CLIP_classification_function, add_winobias_metrics
from transformers import CLIPProcessor, CLIPModel
from winobias_cfg import professions, templates


def show_images(images):
    images = [np.array(image) for image in images]
    images = np.concatenate(images, axis=1)
    return Image.fromarray(images)

def prompt_with_template(profession, template):
    profession = profession.lower()
    custom_prompt = template.replace("{{placeholder}}", profession)
    return custom_prompt

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
    mlp=model_types["MLP"](resolution=args.resolution//64)
    unet.set_controlnet(mlp)
    load_model(unet, args.output_dir+'/unet.pth')
    model=StableDiffusionPipeline(vae=vae,text_encoder=text_encoder,tokenizer=tokenizer,unet=unet,scheduler=scheduler,safety_checker=None,feature_extractor=None,requires_safety_checker=False,)
    model=model.to(args.device)
    if args.evaluation_type=="inference_i2p":
        dataloader=get_i2p_data(data_dir=args.train_data_dir, given_prompt=args.prompt, given_concept=args.concept, max_concept_length=100)
    else:
        dataloader=get_test_data(data_dir=args.train_data_dir, given_prompt=args.prompt, given_concept=args.concept, max_concept_length=100)
    if args.evaluation_type=="eval":
        evaluate(model=model, dataloader=dataloader, args=args)
    elif args.evaluation_type=="interpolate":
        evaluate_interpolate(model=model, dataloader=dataloader, args=args)
    elif args.evaluation_type=="winobias":
        evaluate_inference_winobias(model=model, dataloader=dataloader, args=args)
    elif args.evaluation_type=="i2p":
        evaluate_inference_i2p(model=model, dataloader=dataloader, args=args)
    else:
        raise NotImplementedError(args.evaluation_type)

def predict_cond(model, prompt, seed, condition, device, img_size,num_inference_steps=50,interpolator=None, negative_prompt=None):
    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
    concept_vector = None
    if condition is not None:
        mlp_output = model.unet.controlnet(condition, None)  # Assuming `controlnet` is the MLP
        concept_vector = mlp_output.view(mlp_output.size(0), -1).mean(dim=0).detach().cpu()
    output = model(prompt=prompt, height=img_size, width=img_size, num_inference_steps=num_inference_steps, generator=generator, controlnet_cond=condition,controlnet_interpolator=interpolator,negative_prompt=negative_prompt)
    image = output[0][0]
    return image, concept_vector

def evaluate(model, dataloader, args):
    save_image_dir = os.path.join(args.output_dir, args.image_dir)
    save_vector_dir = os.path.join(args.output_dir, args.vector_dir)
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_vector_dir, exist_ok=True)
    images, concept_vectors, seed = [], [], 0
    for prompt, concept in zip(*dataloader):
        for _ in range(args.num_test_samples):
            image, concept_vector = predict_cond(model=model,prompt=prompt,seed=seed,condition=concept,device=args.device,img_size=args.resolution,num_inference_steps=args.num_inference_steps,negative_prompt=args.negative_prompt)
            images.append(image)
            if concept_vector is not None : concept_vectors.append(concept_vector)
            seed += 1
    images = show_images(images)
    images.save(f"{save_image_dir}/eval.jpg")
    concept_vectors = torch.stack(concept_vectors)
    torch.save(concept_vectors, f"{save_vector_dir}/concept_vectors.pt")
    avg_concept_vector = concept_vectors.mean(dim=0)
    torch.save(avg_concept_vector, f"{save_vector_dir}/avg_concept_vector.pt")

def evaluate_interpolate(model, dataloader, args):
    save_image_dir=args.output_dir+'/'+args.image_dir
    os.makedirs(save_image_dir, exist_ok=True)
    for j in range(args.num_test_samples):
        images=[]
        seed=j
        for prompt, concept in zip(*dataloader):
            if concept is not None:
                for z in np.linspace(0,1,11) : images.append(predict_cond(model=model, prompt=prompt, seed=seed, condition=concept, device = args.device, img_size=args.resolution, interpolator=lambda x,y: x+y*z,num_inference_steps=args.num_inference_steps,negative_prompt=args.negative_prompt))
            else: images.append(predict_cond(model=model, prompt=prompt, seed=seed, condition=None, device = args.device, img_size=args.resolution,num_inference_steps=args.num_inference_steps,negative_prompt=args.negative_prompt))
        images=show_images(images)
        images.save(f"{save_image_dir}/inter{j}.jpg")

def evaluate_inference_i2p(model, dataloader, args):
    seed=None
    save_image_dir=args.output_dir+'/'+args.image_dir
    os.makedirs(save_image_dir, exist_ok=True)
    labels=[]
    model.set_progress_bar_config(disable=True)
    global_id=0
    for j in range(args.num_test_samples):
        for prompt, concept, concept_str in tqdm(dataloader, total=len(dataloader)*(args.num_test_samples-j)):
            image=predict_cond(model=model, prompt=prompt, seed=seed, condition=concept, device = args.device, img_size=args.resolution,num_inference_steps=args.num_inference_steps,negative_prompt=args.negative_prompt)
            image.save(f"{save_image_dir}/{global_id}.jpg")
            global_id+=1
            labels.append([prompt, concept_str])
    prompts, labels = list(zip(*labels))
    predictions=detect_nude_and_q16(folder=save_image_dir)
    logging=pd.DataFrame({'prompt':prompts, 'label':labels, 'prediction':predictions})
    logging.to_csv(save_image_dir + '/i2p.csv', index=False)
    stats=pd.read_csv(save_image_dir + '/i2p.csv').groupby("label").prediction.mean()
    stats.to_csv(save_image_dir + '/i2p_stats.csv', index=True)

def evaluate_inference_winobias(model, dataloader, args):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(args.device)
    seed=None
    logging = []
    root_dir = os.path.join(args.output_dir, 'winobias')
    root_dir = os.path.join(*[root_dir, args.image_dir])
    root_dir = os.path.join(root_dir, f'template{str(args.template_key)}')
    temp_dir = os.path.join(args.output_dir, 'winobias', 'temp')
    print(f'images saved to: {root_dir}')
    for profession in tqdm(professions, desc="Winobias Professions"):  
        save_image_dir = os.path.join(root_dir, profession)
        os.makedirs(save_image_dir, exist_ok=True)
        global_id=0
        template_lst = templates[args.template_key]
        prompts = [prompt_with_template(profession, temp) for temp in template_lst]
        log_file = os.path.join(temp_dir, f"{profession}_log.txt")
        with open(log_file,'w') as temp_file:
            for prompt in prompts:
                print(f'creating images with prompt: {prompt}')
                for _ in range(args.num_test_samples):
                    for _, concept in zip(*dataloader):
                        if concept is not None:
                            image=predict_cond(model=model, prompt=prompt, seed=seed, condition=concept, device = args.device,  img_size=args.resolution,num_inference_steps=args.num_inference_steps,negative_prompt=args.negative_prompt)
                            image[0].save(f"{save_image_dir}/{global_id}.jpg")
                            global_id+=1
            df = CLIP_classification_function(save_image_dir, args.clip_attributes, model=clip_model, processor=processor, return_df=True)
            result = {'profession': profession}
            sums = df.sum().to_dict()
            result.update(sums)
            logging.append(result)
            print(result)
            temp_file.write(str(result))
    logging = pd.DataFrame(logging)
    logging = add_winobias_metrics(logging.set_index('profession'))
    save_name = '_'.join([s.replace(' ', '_') for s in args.clip_attributes])
    save_name += '_result.csv'
    save_path = os.path.join(root_dir, save_name)
    logging.to_csv(save_path, index=True)
    print(f'CLIP classification results saved to {save_path}')

if __name__ == "__main__":
    main()