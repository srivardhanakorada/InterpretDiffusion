import argparse

def parse_args():

    #Initialization
    parser = argparse.ArgumentParser(description="Training Script")

    # Model and training setup
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Data handling
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)

    # Training configuration
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    
    # Optimizer settings
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)

    # Testing
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--template_key", type=str, default="0")
    parser.add_argument("--concept", nargs='+') 
    parser.add_argument("--clip_attributes", type=str, nargs='+') 
    parser.add_argument("--evaluation_type", type=str, default="eval", choices=['eval','interpolate','winobias','i2p'])
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--vector_dir", type=str, default="vectors")
    parser.add_argument("--prompt_file", type=str, default=None)

    # Unknown parameters
    parser.add_argument("--num_inference_steps", type=int, default=50)  # TODO: Check Again
    parser.add_argument("--max_train_samples", type=int, default=None)  # TODO: Check Again
    parser.add_argument("--max_train_steps", type=int, default=None)  # TODO: Check Again
    parser.add_argument("--num_test_samples", type=int, default=2)  # TODO: Check Again
    parser.add_argument("--num_store_model_steps", type=int, default=1000)  # TODO: Check Again

    # Testing metrics (FID, KID)
    parser.add_argument("--src_img_dir", type=str) 
    parser.add_argument("--gen_img_dir", type=str) 
    parser.add_argument("--kid_subset_size", type=int, default=1000)

    # Safety or model modifications
    parser.add_argument("--use_sld", action='store_true')
    parser.add_argument("--use_esd", action='store_true')
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default='pndm')

    # Finale
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)