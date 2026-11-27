import os
import torch
import numpy as np

from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel
from random import randrange

from src.mgd_pipelines.mgd_pipe import MGDPipe
from src.datasets.vitonhd import VitonHDDataset

# EXPERIMENT SETTINGS

NUM_VARIATIONS = 20

NUM_INFERENCE_STEPS = 10

GUIDANCE_SCALE = 7.5

BASE_SEED = 1000

# SKETCH CONDITIONING

SKETCH_COND_RATE = 0.6
START_COND_RATE = 0

# SETTINGS

SD_PATH = r"C:\mgd_models\stable-diffusion-inpainting"

MGD_WEIGHTS = r"C:\mgd_models\dresscode.pth"

DATA_ROOT = (
    r"C:\Users\osun2\IdeaProjects"
    r"\multimodal-garment-designer-fork"
    r"\assets\data\vitonhd"
)

OUTPUT_DIR = "dataset_exploration"

# OUTPUT DIRECTORY

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# LOAD MGD UNET

print("Loading MGD UNet...")

config = UNet2DConditionModel.load_config(
    SD_PATH + "/unet"
)

config["in_channels"] = 28

unet = UNet2DConditionModel.from_config(
    config
)

unet.load_state_dict(
    torch.load(
        MGD_WEIGHTS,
        map_location="cpu"
    )
)

unet.eval()

print("MGD UNet loaded!")

# LOAD STABLE DIFFUSION

print("Loading Stable Diffusion components...")

text_encoder = CLIPTextModel.from_pretrained(
    SD_PATH,
    subfolder="text_encoder",
    local_files_only=True
)

vae_config = AutoencoderKL.load_config(
    SD_PATH + "/vae"
)

vae = AutoencoderKL.from_config(
    vae_config
)

vae.load_state_dict(
    torch.load(
        SD_PATH + "/vae/diffusion_pytorch_model.bin",
        map_location="cpu"
    )
)

tokenizer = CLIPTokenizer.from_pretrained(
    SD_PATH,
    subfolder="tokenizer",
    local_files_only=True
)

scheduler = DDIMScheduler.from_pretrained(
    SD_PATH,
    subfolder="scheduler",
    local_files_only=True
)

scheduler.set_timesteps(
    NUM_INFERENCE_STEPS
)

# CREATE MGD PIPELINE

print("Creating MGD pipeline...")

mgd_pipe = MGDPipe(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=scheduler,
)

print("MGD PIPELINE LOADED!")

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


def run_var():
    # SELECT PERSON

    person_index = randrange(len(dataset))
    # person_index = 1952
    sample = dataset[person_index]

    print("Using person sample:", person_index)

    print(
        "Image:", sample["im_name"]
    )

    # INPUTS

    image = sample["image"].unsqueeze(0)
    pose_map = sample["pose_map"].unsqueeze(0)
    mask_image = sample["inpaint_mask"].unsqueeze(0)

    # Because order="unpaired", this is the
    # im_sketch_unpaired image internally.
    sketch = sample["im_sketch"].unsqueeze(0)

    # SAVE INPUTS

    original_np = (image[0].permute(1, 2, 0).cpu().numpy())
    original_np = (original_np + 1) / 2 * 255
    original_np = np.clip(original_np, 0, 255).astype(np.uint8)

    Image.fromarray(original_np).save(
        os.path.join(
            OUTPUT_DIR,
            f"original_person_{person_index}.png"
        )
    )

    # MASK

    mask_debug = mask_image[0].cpu()

    if mask_debug.ndim == 3:
        mask_debug = mask_debug.squeeze(0)

    mask_debug = mask_debug.numpy() * 255

    mask_debug = np.clip(mask_debug, 0, 255).astype(np.uint8)

    Image.fromarray(mask_debug).save(
        os.path.join(
            OUTPUT_DIR,
            f"mask_{person_index}.png"
        )
    )

    # SKETCH

    sketch_debug = sketch[0].cpu()

    if sketch_debug.shape[0] == 1:
        sketch_debug = sketch_debug.squeeze(0)

    sketch_debug = sketch_debug.numpy() * 255

    sketch_debug = np.clip(sketch_debug, 0, 255).astype(np.uint8)

    Image.fromarray(sketch_debug).save(
        os.path.join(
            OUTPUT_DIR,
            f"garment_sketch_{person_index}.png"
        )
    )

    # ORIGINAL IMAGE FOR COMPOSITING

    original_np = image[0].permute(1, 2, 0).cpu().numpy()
    original_np = (original_np + 1) / 2 * 255
    original_np = np.clip(original_np, 0, 255)

    # MASK FOR COMPOSITING

    mask_np = mask_image[0].cpu().numpy()

    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze(0)

    mask_np = mask_np[:, :, None]

    print("Person Index:", person_index)

    # SET RANDOM SEED

    torch.manual_seed(BASE_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_SEED)

    # GENERATE
    print("Starting generation...")

    result = mgd_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        pose_map=pose_map,
        sketch=sketch,
        height=512,
        width=384,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        sketch_cond_rate=SKETCH_COND_RATE,
        start_cond_rate=START_COND_RATE,
        output_type="pil",
    )

    print("Generation complete!")

    generated = result.images[0]

    # SAVE RAW OUTPUT
    raw_filename = f"var_{person_index}_raw.png"
    raw_path = os.path.join(OUTPUT_DIR, raw_filename)

    print("Saving raw image to:", raw_path)
    generated.save(raw_path)

    print("Raw image saved!")

    # COMPOSITE
    generated_np = np.array(generated).astype(np.float32)
    final_np = generated_np * mask_np + original_np * (1 - mask_np)
    final_np = np.clip(final_np, 0, 255).astype(np.uint8)

    final_image = Image.fromarray(final_np)

    # SAVE FINAL OUTPUT
    filename = f"var_{person_index:02d}.png"
    final_path = os.path.join(OUTPUT_DIR, filename)

    print("Saving final image to:", final_path)
    final_image.save(final_path)

    print("Final image saved!")


# TEXT PROMPT

prompt = ("a garment featuring multiple complementary and contrasting colors distributed across its design "
          "with varied textile treatments and intentional surface designs, including printed motifs, "
          "geometric patterns, stripes, and decorative details")
negative_prompt = ("dense fine speckling, tiny evenly distributed dots, stippled noise, artificial micro-dot "
                   "patterns, fine repetitive pixel-like markings covering the fabric")
# START EXPERIMENT

print("Number of variations:", NUM_VARIATIONS)
print("Prompt:", prompt)
print("Sketch conditioning:", SKETCH_COND_RATE)
print("Inference steps:", NUM_INFERENCE_STEPS)
print("Base seed:", BASE_SEED)

for i in range(NUM_VARIATIONS):
    run_var()

# DONE

print("Results saved to:", OUTPUT_DIR)

print()
print("Generated", NUM_VARIATIONS, "variations.")
