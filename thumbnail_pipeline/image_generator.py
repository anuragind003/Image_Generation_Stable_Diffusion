# image_generator.py
import torch
from diffusers import DiffusionPipeline
import os
from config import (
    SD_BASE_MODEL_ID,
    # VAE_MODEL_ID, # Uncomment if using
)

# Global variable to hold the pipeline to avoid reloading
pipeline = None

def load_sd_pipeline(lora_path, base_model_id):
    """Loads the SDXL pipeline and applies LoRA weights."""
    global pipeline
    if pipeline is not None:
        print("SD Pipeline already loaded.")
        return pipeline

    print(f"Loading SDXL Base Model: {base_model_id}...")
    try:
        # Determine data type based on GPU availability/type
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # # Optional: Load VAE if specified
        # vae = None
        # if VAE_MODEL_ID:
        #     try:
        #         from diffusers import AutoencoderKL
        #         print(f"Loading VAE: {VAE_MODEL_ID}")
        #         vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, torch_dtype=dtype)
        #         print("VAE loaded.")
        #     except Exception as e:
        #         print(f"Warning: Could not load VAE {VAE_MODEL_ID}. Error: {e}")
        #         vae = None

        pipeline = DiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            # variant="fp16", # Use if available and appropriate
            use_safetensors=True,
            # vae=vae # Pass VAE if loaded
        )
        print("Base model loaded.")

        # Load LoRA weights
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA weights from: {lora_path}")
            # The exact method might depend slightly on diffusers version
            # Common methods:
            # 1. Directly loading into the pipeline
            pipeline.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path))
            # 2. If using PEFT adapters (less common for simple LoRA file):
            # pipeline.load_adapter(lora_path, adapter_name="custom_lora")
            # pipeline.set_adapters(["custom_lora"], adapter_weights=[1.0]) # If needed
            print("LoRA weights loaded successfully.")
        else:
            print(f"Warning: LoRA path '{lora_path}' not found or not specified. Generating without LoRA.")


        # Move to GPU if available
        if torch.cuda.is_available():
            print("Moving pipeline to GPU...")
            pipeline.to("cuda")
            print("Pipeline on GPU.")
        else:
            print("Warning: CUDA not available. Running on CPU (will be very slow).")

        # Optional: Enable memory-saving optimizations if needed/available
        # pipeline.enable_model_cpu_offload() # If VRAM is extremely limited
        # pipeline.enable_xformers_memory_efficient_attention() # If xformers is installed

        print("SD Pipeline initialized.")
        return pipeline

    except Exception as e:
        print(f"Error loading Stable Diffusion pipeline: {e}")
        pipeline = None # Ensure pipeline is None if loading failed
        return None


def generate_image(pipe, prompt, negative_prompt, output_path, steps, guidance_scale, seed=None):
    """Generates an image using the loaded pipeline."""
    if pipe is None:
        print("Error: Pipeline is not loaded.")
        return None

    print(f"\n--- Generating Image ---")
    print(f"Prompt: {prompt}")
    # print(f"Negative Prompt: {negative_prompt}") # Debugging
    print(f"Steps: {steps}, Guidance: {guidance_scale}, Seed: {seed}")
    print(f"Output Path: {output_path}")

    try:
        generator = None
        if seed is not None:
             # Ensure seed is integer
             try:
                 seed = int(seed)
                 if torch.cuda.is_available():
                     generator = torch.Generator("cuda").manual_seed(seed)
                 else:
                     generator = torch.Generator("cpu").manual_seed(seed)
             except ValueError:
                  print(f"Warning: Invalid seed value '{seed}'. Using random seed.")
                  seed = None # Fallback to random

        # Generate the image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        # Save the image
        image.save(output_path)
        print(f"Image saved successfully to {output_path}")
        return output_path

    except Exception as e:
        print(f"Error during image generation: {e}")
        # Consider adding torch.cuda.empty_cache() here if it's an OOM error
        if "CUDA out of memory" in str(e):
            print("CUDA OOM Error detected. Clearing cache...")
            torch.cuda.empty_cache()
        return None

def unload_sd_pipeline():
    """Releases the SD pipeline from memory (GPU)."""
    global pipeline
    if pipeline is not None:
        print("Unloading SD pipeline from memory...")
        # Explicitly delete references and clear cache
        del pipeline
        pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Pipeline unloaded.")