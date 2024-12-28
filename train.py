import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from model import model_types
from config import parse_args
from utils.utils_model import save_model
from utils.utils_data import get_dataloader


def main():
    
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError()
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids
    
    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example[1] for example in examples]
        padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        input_conditions = torch.stack([example[2] for example in examples])
        return {"pixel_values": pixel_values,"input_ids": padded_tokens.input_ids,"attention_mask": padded_tokens.attention_mask,"input_conditions": input_conditions,}
    
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    mlp=model_types["MLP"](resolution=args.resolution//64)
    unet.set_controlnet(mlp) ## Opposed to element wise addition in the paper
    
    optimizer = torch.optim.Adam(unet.parameters(),lr=args.learning_rate,betas=(args.adam_beta1, args.adam_beta2),weight_decay=args.adam_weight_decay,eps=args.adam_epsilon)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    train_transforms = transforms.Compose([transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR), transforms.RandomCrop(args.resolution),transforms.RandomHorizontalFlip(), transforms.Lambda(lambda x: x),transforms.ToTensor(),transforms.Normalize([0.5], [0.5]),])
    train_dataloader = get_dataloader(args.train_data_dir, batch_size=args.train_batch_size, shuffle=True,transform=train_transforms, tokenizer=tokenize_captions, collate_fn=collate_fn,num_workers=4, max_concept_length=100, select="random")
    num_update_steps_per_epoch = len(train_dataloader) ## Since the custom dataloader already divides into batches
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler("constant",optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.max_train_steps)
    weight_dtype = torch.float32
    text_encoder.to(args.device, dtype=weight_dtype)
    vae.to(args.device, dtype=weight_dtype)
    unet.to(args.device,dtype=weight_dtype) ## Added dtype
    progress_bar = tqdm(range(args.max_train_steps))
    progress_bar.set_description("Steps")
    loss_history=[]
    train_loss = 0.0
    global_step = 0

    for _ in range(args.num_train_epochs):
        unet.train()
        for _, batch in enumerate(train_dataloader):
            latents = vae.encode(batch["pixel_values"].to(weight_dtype).to(args.device)).latent_dist.sample()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=args.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(batch["input_ids"].to(args.device))[0]
            if noise_scheduler.config.prediction_type == "epsilon": target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction": target = noise_scheduler.get_velocity(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, controlnet_cond=batch["input_conditions"].to(args.device)).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            train_loss = loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            global_step += 1
            loss_history.append(train_loss)
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps: break
            if global_step % args.num_store_model_steps == 0: save_model(unet, args.output_dir+'/unet.pth')
        save_model(unet, args.output_dir+'/unet.pth')

    plt.figure()
    plt.plot(loss_history)
    plt.savefig(args.output_dir+'/loss_history.png')
    plt.close()

if __name__ == "__main__": main()