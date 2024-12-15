import os
import json
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

def create_concept_to_id_mapping():
    concepts = ["female", "young-female", "old-female"]
    return {concept: idx for idx, concept in enumerate(concepts)}

def repeat_elements(elements, n):
    return [element for element in elements for _ in range(n)]

class DatasetGenerator:
    
    def __init__(self, config):
        self.output_dir = config.output_dir
        self.prompts = config.prompts
        self.corrupted_prompts_and_targets = config.corrupted_prompts_and_targets
        self.num_samples = config.num_samples
        print(f"Generating {self.num_samples} samples per concept in {self.output_dir}")

    def generate_images(self, num_inference_steps=30):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe = pipe.to("cuda")
        pipe.safety_checker = None
        pipe.set_progress_bar_config(disable=True)
        for concept_idx, (prompt, corrupted_target) in enumerate(zip(self.prompts, self.corrupted_prompts_and_targets)):
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
        json.dump(create_concept_to_id_mapping(), open(f"{self.output_dir}/concept_dict.json", "w"))

    def run(self):
        self.generate_images()

class DatasetConfig:
    output_dir = "data/female"
    num_samples = 1000
    prompts = [
        "a female doctor",
        "a young-female doctor",
        "a old-female doctor"
    ]
    corrupted_prompts_and_targets = [
        [["a doctor", ["female"]]],
        [["a doctor", ["young-female"]]],
        [["a doctor", ["old-female"]]],
    ]

generator = DatasetGenerator(DatasetConfig)
generator.run()