
# config_trainer.py
import os

# --- Project Paths on Google Drive ---
# Base directory where data/outputs reside (NOT where the code is)
# Adjust this path if your data folders are elsewhere
DRIVE_PROJECT_BASE = "/content/drive/MyDrive/db_lora_sdxl_project"

INSTANCE_DATA_DIR = os.path.join(DRIVE_PROJECT_BASE, "instance_images")
CLASS_DATA_DIR = os.path.join(DRIVE_PROJECT_BASE, "class_images") # Used if prior preservation is enabled
OUTPUT_DIR = os.path.join(DRIVE_PROJECT_BASE, "output_lora")

# --- Diffusers Repo / Training Script Location ---
# Where to clone the diffusers repo
DIFFUSERS_REPO_DIR = "/content/diffusers"
# Path to the specific training script within the cloned repo
TRAIN_SCRIPT_PATH = os.path.join(DIFFUSERS_REPO_DIR, "examples/dreambooth/train_dreambooth_lora_sdxl.py")
TRAIN_SCRIPT_DIR = os.path.dirname(TRAIN_SCRIPT_PATH) # /content/diffusers/examples/dreambooth

# --- Model Configuration ---
PRETRAINED_MODEL_NAME_OR_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
# PRETRAINED_VAE_NAME_OR_PATH = "madebyollin/sdxl-vae-fp16-fix" # Optional VAE

# --- LoRA Instance/Class Prompts ---
# <<< CHANGE THESE >>>
UNIQUE_TOKEN = "kuku"
CLASS_NAME = "thumbnail style"
# <<< CHANGE THESE >>>

INSTANCE_PROMPT = f"a photo in {UNIQUE_TOKEN} {CLASS_NAME}"
CLASS_PROMPT = f"a photo in {CLASS_NAME}" # Used only if prior preservation enabled

# --- Training Hyperparameters ---
# Adjust MAX_TRAIN_STEPS based on dataset size (e.g., 10-15 steps per instance image)
MAX_TRAIN_STEPS = 600  # Example: Reduced for ~33 images, maybe 1000-1500 for 100 images
LEARNING_RATE = 1e-4   # Or try 5e-5 if overfitting
LR_SCHEDULER = "constant"
LR_WARMUP_STEPS = 0
LORA_RANK = 8

# --- Batching and Memory ---
# For Free Colab T4 GPU: batch_size=1, grad_accum=4 is usually max possible for SDXL
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = "fp16"  # 'fp16' or 'bf16' (T4 prefers fp16)
USE_8BIT_ADAM = True
GRADIENT_CHECKPOINTING = True
ENABLE_XFORMERS = True  # Set to False if install fails or causes issues

# --- Validation and Saving ---
SAVE_STEPS = 250  # How often to save LoRA checkpoints
VALIDATION_PROMPT = f"{INSTANCE_PROMPT}, cinematic lighting, high detail"
VALIDATION_EPOCHS = 50 # How often (in epochs) to run validation/save samples. May cause OOM.

# --- Other Parameters ---
SEED = 42
RESOLUTION = 1024 # SDXL native resolution
REPORT_TO = "tensorboard" # Could be "wandb" if configured

# --- Prior Preservation Control ---
# Set to True to enable prior preservation (requires images in CLASS_DATA_DIR)
WITH_PRIOR_PRESERVATION = True # <<< SET TO True or False >>>
PRIOR_LOSS_WEIGHT = 1.0
# num_class_images will be calculated automatically if enabled

# Print paths for verification when loaded
print(f"ConfigTrainer - Instance Dir: {INSTANCE_DATA_DIR}")
print(f"ConfigTrainer - Class Dir: {CLASS_DATA_DIR}")
print(f"ConfigTrainer - Output Dir: {OUTPUT_DIR}")
print(f"ConfigTrainer - Train Script Dir: {TRAIN_SCRIPT_DIR}")