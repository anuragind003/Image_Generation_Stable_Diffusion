# config.py
import os
from dotenv import load_dotenv # Optional: For loading API key from .env file

# --- API Keys (IMPORTANT: Use environment variables or secure methods) ---
load_dotenv() # Load .env file if it exists
# --- Make sure your key is set here or in GOOGLE_API_KEY environment variable ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "GEMINI_API_HERE")
# HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None) # If needed for private models

# --- Model IDs ---
SD_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix" # Optional

# --- LoRA Details ---
# Assumes LoRA file is in the same directory or a sub-directory
# Adjust the path as needed relative to where you run main_pipeline.py
# --- Make sure this path is correct ---
LORA_FILE_PATH = "/content/drive/MyDrive/db_lora_sdxl_project/output_lora/pytorch_lora_weights.safetensors"
# --- Make sure this matches your training trigger ---
LORA_TRIGGER_PHRASE = "kuku thumbnail style"

# --- Generation Parameters ---
NUM_VARIATIONS = 3 # How many distinct thumbnails to generate
SD_DEFAULT_STEPS = 30
SD_DEFAULT_GUIDANCE_SCALE = 7.5
NEGATIVE_PROMPT_DEFAULT = "blurry, low quality, worst quality, text, signature, watermark, username, words, disfigured, deformed, duplicate"

# --- File Paths ---
# Define output directories relative to the main script or use absolute paths
# Ensure these directories exist or are created by the script
OUTPUT_DIR = "/content/drive/MyDrive/db_lora_sdxl_project/generated_thumbnails" # Main output
TEMP_DIR = "/content/temp_thumbnails" # For intermediate images before text
# --- CORRECTED FONT DIRECTORY PATH ---
FONT_DIR = "/content/drive/MyDrive/thumbnail_pipeline/fonts" # Directory containing .ttf/.otf files

# --- Font Configuration (MUST match files in FONT_DIR) ---
FONT_MAPPING = {
    # --- Ensure 'DejaVuSans-Bold.ttf' exists in the FONT_DIR above ---
    "sans-serif": os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"),
    # --- If using 'tech', ensure 'Orbitron-Bold.ttf' exists in FONT_DIR ---
    # "tech": os.path.join(FONT_DIR, "Orbitron-Bold.ttf"),
    # Add more font mappings as needed, ensuring files exist
}
# --- Make sure this key exists in FONT_MAPPING above ---
DEFAULT_FONT_KEY = "sans-serif"

# --- Text Overlay Configuration ---
DEFAULT_FONT_SIZE = 70
DEFAULT_TEXT_COLOR = (255, 255, 255) # White
DEFAULT_OUTLINE_COLOR = (0, 0, 0) # Black
DEFAULT_OUTLINE_WIDTH = 2
DEFAULT_PLACEMENT_TEMPLATE = "bottom_center" # Options: "bottom_center", "bottom_left", etc.

# --- Ensure default font path is valid (REVISED LOGIC from previous step) ---
DEFAULT_FONT_PATH = None # Initialize as None (invalid)

print("\n--- Verifying Font Configuration ---")
print(f"FONT_DIR configured as: {FONT_DIR}") # This will now print the correct path
if not os.path.isdir(FONT_DIR):
    print(f"CRITICAL ERROR: Font directory '{FONT_DIR}' does not exist!")
else:
    print(f"Font directory exists.")
    # --- Check the DEFAULT_FONT_KEY first ---
    if DEFAULT_FONT_KEY in FONT_MAPPING:
        potential_default_path = FONT_MAPPING[DEFAULT_FONT_KEY]
        print(f"Checking default key '{DEFAULT_FONT_KEY}' pointing to: {potential_default_path}")
        if os.path.exists(potential_default_path):
            DEFAULT_FONT_PATH = potential_default_path
            print(f"SUCCESS: Default font '{DEFAULT_FONT_KEY}' found and path is valid: {DEFAULT_FONT_PATH}")
        else:
            print(f"WARNING: Default font key '{DEFAULT_FONT_KEY}' exists in mapping, but path is invalid: {potential_default_path}")
    else:
        print(f"WARNING: DEFAULT_FONT_KEY ('{DEFAULT_FONT_KEY}') not found in FONT_MAPPING keys.")

    # --- If default path is STILL None, try finding ANY valid font ---
    if DEFAULT_FONT_PATH is None:
        print("Default font path invalid or not found. Searching for any valid font in FONT_MAPPING...")
        found_valid_alternative = False
        for key, path in FONT_MAPPING.items():
            print(f"Checking alternative key '{key}' pointing to: {path}")
            if os.path.exists(path):
                DEFAULT_FONT_PATH = path
                print(f"SUCCESS: Found valid alternative font '{key}' to use as default: {DEFAULT_FONT_PATH}")
                found_valid_alternative = True
                break # Stop searching once one is found
            else:
                 print(f"Path for key '{key}' is invalid: {path}")

        if not found_valid_alternative:
            print("CRITICAL ERROR: No valid font paths found in any FONT_MAPPING entry.")
            # DEFAULT_FONT_PATH remains None

# --- Final Check ---
if DEFAULT_FONT_PATH is None:
    print("CONFIGURATION ISSUE: DEFAULT_FONT_PATH could not be set. Text overlay will fail.")
else:
    print(f"Final DEFAULT_FONT_PATH set to: {DEFAULT_FONT_PATH}")
print("--- End Font Configuration Verification ---\n")


# Print key paths for verification (keep these)
print(f"Config - Font Dir: {FONT_DIR}")
print(f"Config - Lora Path: {LORA_FILE_PATH}")
print(f"Config - Output Dir: {OUTPUT_DIR}")