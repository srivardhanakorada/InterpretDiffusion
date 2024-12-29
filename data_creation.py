import os
import json
from tqdm import tqdm
import argparse
from diffusers import StableDiffusionPipeline

def load_config(config_path):
    with open(config_path, 'r') as file: return json.load(file)

def create_concept_to_id_mapping(concepts): return {concept: idx for idx, concept in enumerate(concepts)}

def repeat_elements(elements, n): return [element for element in elements for _ in range(n)]

class DatasetGenerator:
    
    def __init__(self, config):
        self.output_dir = config["output_dir"]
        self.prompts = config["prompts"]
        self.corrupted_prompts_and_targets = config["corrupted_prompts_and_targets"]
        self.validation_prompts = config["validation_prompts"]
        self.num_samples = config["num_samples"]
        self.device = config["device"]
        print(f"Generating {self.num_samples} samples per concept in {self.output_dir}")

    def generate_images(self, num_inference_steps=30):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe = pipe.to(self.device)
        pipe.safety_checker = None
        pipe.set_progress_bar_config(disable=True)
        for concept_idx, (prompt, corrupted_target, validation_prompt) in enumerate(zip(self.prompts, self.corrupted_prompts_and_targets, self.validation_prompts)):
            concept_dir = os.path.join(self.output_dir, f"concept_{concept_idx}")
            os.makedirs(concept_dir, exist_ok=True)
            print(f"Generating images for concept {concept_idx}: '{prompt}' in {concept_dir}")
            expanded_prompts = repeat_elements([prompt], self.num_samples)
            expanded_targets = repeat_elements([corrupted_target], self.num_samples)
            for idx, p in tqdm(enumerate(expanded_prompts), total=len(expanded_prompts), desc=f"Concept {concept_idx}"):
                output = pipe(p, num_inference_steps=num_inference_steps, return_dict=True)
                image = output.images[0]
                image.save(f"{concept_dir}/{idx}.jpg")
            json.dump(expanded_targets, open(f"{concept_dir}/labels.json", "w"))
            json.dump(validation_prompt, open(f"{concept_dir}/test.json", "w"))
        concepts = [target[0][1][0] for target in self.corrupted_prompts_and_targets]
        json.dump(create_concept_to_id_mapping(concepts), open(f"{self.output_dir}/concept_dict.json", "w"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)
    DatasetGenerator(config).generate_images()