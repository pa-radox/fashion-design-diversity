
import os
import torch
from PIL import Image
from diffusers import (
    StableDiffusionInpaintPipeline,
    DDIMScheduler
)
from transformers import CLIPTokenizer
from random import randrange
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)
from src.datasets.vitonhd import VitonHDDataset

# EXPERIMENT SETTINGS

NUM_VARIATIONS = 3

NUM_INFERENCE_STEPS = 20

GUIDANCE_SCALE = 7.5

BASE_SEED = 1000


# PATHS


SD_PATH = r"C:\mgd_models\stable-diffusion-inpainting"

DATA_ROOT = (
    r"C:\Users\osun2\IdeaProjects"
    r"\multimodal-garment-designer-fork"
    r"\assets\data\vitonhd"
)

OUTPUT_DIR = "inpainting_baseline"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# TEXT PROMPT

prompt = (
    "a fashionable garment with distinctive and unique design details."
)

# LOAD TOKENIZER

print("Loading tokenizer...")

tokenizer = CLIPTokenizer.from_pretrained(
    SD_PATH,
    subfolder="tokenizer",
    local_files_only=True
)

# LOAD VITON-HD

print("Loading VITON-HD dataset...")

dataset = VitonHDDataset(
    dataroot_path=DATA_ROOT,
    phase="test",
    tokenizer=tokenizer,
    order="unpaired",

    outputlist=(
        "im_name",
        "image",
        "shape",
        "pose_map",
        "im_mask",
        "inpaint_mask",
        "parse_mask_total",
        "im_sketch"
    ),

    size=(512, 384),
)

print("Dataset loaded!")
print("Dataset size:", len(dataset))

# SELECT PERSON

# PERSON_INDEX = randrange(len(dataset))
PERSON_INDEX = 1952
sample = dataset[PERSON_INDEX]

print("Using person sample:", PERSON_INDEX)
print("Image:", sample["im_name"])

# PREPARE INPUTS

person_tensor = sample["image"]

sketch_tensor = sample["im_sketch"]

# Convert person image from [-1, 1] to [0, 255]
person_np = person_tensor.permute(1, 2, 0).cpu().numpy()

person_np = (person_np + 1) / 2 * 255
person_np = person_np.clip(0, 255).astype("uint8")

person_image = Image.fromarray(person_np).convert("RGB")

# PREPARE MASK

mask_tensor = sample["inpaint_mask"]

# Convert mask from [0, 1] to [0, 255]
mask_np = mask_tensor.squeeze().cpu().numpy()

mask_np = mask_np.clip(0, 1) * 255
mask_np = mask_np.astype("uint8")

mask_image = Image.fromarray(mask_np).convert("L")

# Save sketch for documentation.
# NOTE: Standard Inpainting does NOT receive this sketch.
sketch_np = sketch_tensor.squeeze().cpu().numpy()

sketch_np = sketch_np.clip(0, 1) * 255
sketch_np = sketch_np.astype("uint8")

sketch_image = Image.fromarray(sketch_np).convert("L")

# SAVE EXPERIMENT INPUTS

person_image.save(
    os.path.join(OUTPUT_DIR, "original_person.png")
)

mask_image.save(
    os.path.join(OUTPUT_DIR, "mask.png")
)

sketch_image.save(
    os.path.join(OUTPUT_DIR, "garment_sketch.png")
)

# LOAD STABLE DIFFUSION INPAINTING

print("Loading Stable Diffusion Inpainting...")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    SD_PATH,
    torch_dtype=torch.float32,
    local_files_only=True
)

# Use DDIM, matching the scheduler used by MGD
pipe.scheduler = DDIMScheduler.from_pretrained(
    SD_PATH,
    subfolder="scheduler",
    local_files_only=True
)

pipe = pipe.to("cpu")

pipe.enable_attention_slicing()

print("Stable Diffusion Inpainting loaded!")

# EXPERIMENT INFORMATION

print("Number of variations:", NUM_VARIATIONS)
print("Prompt:", prompt)
print("Inference steps:", NUM_INFERENCE_STEPS)
print("Guidance scale:", GUIDANCE_SCALE)
print("Base seed:", BASE_SEED)
print("Person index:", PERSON_INDEX)

print()
print("Generating on CPU. This may take a while.")
print()

# GENERATE VARIATIONS

for i in range(NUM_VARIATIONS):

    variation_number = i + 1
    seed = BASE_SEED + i

    print("----------------------------------------")
    print(
        f"VARIATION {variation_number}/{NUM_VARIATIONS}"
    )
    print("Seed:", seed)

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)

    print("Starting generation...")

    result = pipe(
        prompt=prompt,

        image=person_image,

        mask_image=mask_image,

        height=512,
        width=384,

        num_inference_steps=NUM_INFERENCE_STEPS,

        guidance_scale=GUIDANCE_SCALE,

        generator=generator
    ).images[0]

    print("Generation complete!")

    # SAVE OUTPUT
    filename = f"var_{variation_number:02d}.png"

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    result.save(output_path)

    print("Saved:", output_path)

# DONE

print("Results saved to:", OUTPUT_DIR)
print("Generated", NUM_VARIATIONS, "variations.")
