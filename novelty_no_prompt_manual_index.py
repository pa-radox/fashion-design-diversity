import os
import random

import torch
import numpy as np

from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from src.mgd_pipelines.mgd_pipe import MGDPipe
from src.datasets.vitonhd import VitonHDDataset
import torchvision.transforms.functional as tf

# EXPERIMENT SETTINGS

NUM_VARIATIONS = 5

NUM_INFERENCE_STEPS = 30

GUIDANCE_SCALE = 7.5

BASE_SEED = 1000

PERSON_INDEXES = [2, 414, 456, 482, 1131, 1200, 1321, 1873]

# SKETCH CONDITIONING

SKETCH_COND_RATE = 0.6
START_COND_RATE = 0

# SETTINGS

SD_PATH = r"C:\mgd_models\stable-diffusion-inpainting"

# MGD_WEIGHTS = r"C:\mgd_models\vitonhd.pth"

DATA_ROOT = (
    r"C:\Users\osun2\IdeaProjects"
    r"\multimodal-garment-designer-fork"
    r"\assets\data\vitonhd"
)

OUTPUT_DIR = "novelty_results2"

# OUTPUT DIRECTORY

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# LOAD MGD UNET

print("Loading MGD UNet...")

unet = torch.hub.load(
    repo_or_dir="aimagelab/multimodal-garment-designer",
    source="github",
    model="mgd",
    dataset="vitonhd",
    pretrained=True
)

unet = unet.to("cpu")
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


def run_var(person_index, var_num):
    # SELECT PERSON
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

    print("Person Index:", person_index)

    # SET RANDOM SEED
    seed = BASE_SEED + var_num
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        generator=generator,
        output_type="pil",
    )

    print("Generation complete!")

    generated = result.images[0]

    # SAVE RAW OUTPUT
    raw_filename = f"var_{person_index}_{var_num}_raw.png"
    raw_path = os.path.join(OUTPUT_DIR, raw_filename)

    print("Saving raw image to:", raw_path)
    generated.save(raw_path)

    print("Raw image saved!")

    # # COMPOSITE
    # generated_np = np.array(generated).astype(np.float32)
    # final_np = generated_np * mask_np + original_np * (1 - mask_np)
    # final_np = np.clip(final_np, 0, 255).astype(np.uint8)
    #
    # final_image = Image.fromarray(final_np)
    #
    # # SAVE FINAL OUTPUT
    # filename = f"var_{variation_number:02d}.png"
    # final_path = os.path.join(OUTPUT_DIR, filename)
    #
    # print("Saving final image to:", final_path)
    # final_image.save(final_path)
    #
    # print("Final image saved!")

    # Original image, converted from [-1, 1] to [0, 1]
    gt_img = image[0] * 0.5 + 0.5

    # Get original human parsing map
    im_parse = sample["im_parse"]

    # Identify head pixels
    seg_head = (
            (im_parse == 1) |
            (im_parse == 2) |
            (im_parse == 4) |
            (im_parse == 13)
    )
    true_head = gt_img * seg_head

    # Use generated image everywhere except the head
    fake_img = tf.pil_to_tensor(generated).float() / 255.0
    generated_body = fake_img * (~seg_head)

    final_img = true_head + generated_body
    final_img = final_img.clamp(0, 1)

    final_filename = f"var_{person_index}_{var_num}_final.png"
    final_path = os.path.join(
        OUTPUT_DIR,
        final_filename
    )
    tf.to_pil_image(final_img).save(final_path)


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

for index in PERSON_INDEXES:
    for i in range(NUM_VARIATIONS):
        run_var(index, i)

# DONE

print("Results saved to:", OUTPUT_DIR)

print()
print("Generated", NUM_VARIATIONS, "variations.")
