# main_pipeline.py
import os
import shutil
import random
import time

# Import configurations and functions from other modules
import config
import llm_handler
import image_generator
import text_overlay

def create_directories():
    """Creates necessary output and temporary directories."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    os.makedirs(config.FONT_DIR, exist_ok=True) # Ensure font dir exists too
    print(f"Ensured directories exist: {config.OUTPUT_DIR}, {config.TEMP_DIR}")

def cleanup_temp_files():
    """Removes the temporary image directory."""
    if os.path.exists(config.TEMP_DIR):
        try:
            shutil.rmtree(config.TEMP_DIR)
            print(f"Cleaned up temporary directory: {config.TEMP_DIR}")
        except Exception as e:
            print(f"Warning: Could not remove temporary directory {config.TEMP_DIR}. Error: {e}")

def run_thumbnail_pipeline(user_title, user_concept_note):
    """Orchestrates the full thumbnail generation pipeline."""
    start_time = time.time()
    create_directories()
    final_thumbnail_paths = []
    temp_image_paths = []

    # --- Step 1: Generate SD Prompts using LLM ---
    print("\n=== Step 1: Generating Stable Diffusion Prompts via Gemini ===")
    sd_prompt_generation_prompt = llm_handler.generate_sd_prompts_prompt(
        user_title,
        user_concept_note,
        config.NUM_VARIATIONS,
        config.LORA_TRIGGER_PHRASE,
        config.NEGATIVE_PROMPT_DEFAULT
    )
    sd_prompts_data = llm_handler.call_gemini(sd_prompt_generation_prompt, expect_json=True)

    if not sd_prompts_data or not isinstance(sd_prompts_data, list) or len(sd_prompts_data) == 0:
        print("Error: Failed to get valid SD prompts from Gemini. Aborting.")
        return []

    print(f"Successfully received {len(sd_prompts_data)} SD prompt variations.")

    # --- Step 2: Load SD Model (Load once) ---
    print("\n=== Step 2: Loading Stable Diffusion Model and LoRA ===")
    sd_pipe = image_generator.load_sd_pipeline(config.LORA_FILE_PATH, config.SD_BASE_MODEL_ID)
    if sd_pipe is None:
        print("Error: Failed to load Stable Diffusion pipeline. Aborting.")
        return []

    # --- Step 3: Generate Base Images ---
    print("\n=== Step 3: Generating Base Images ===")
    for i, prompt_data in enumerate(sd_prompts_data):
        if not isinstance(prompt_data, dict) or "prompt" not in prompt_data or "negative_prompt" not in prompt_data:
            print(f"Warning: Skipping invalid prompt data at index {i}: {prompt_data}")
            continue

        prompt = prompt_data["prompt"]
        negative_prompt = prompt_data["negative_prompt"]
        # Ensure the trigger phrase is definitely in the prompt (Gemini might forget)
        if config.LORA_TRIGGER_PHRASE not in prompt:
             prompt += f", {config.LORA_TRIGGER_PHRASE}" # Append if missing
             print(f"Warning: Trigger phrase '{config.LORA_TRIGGER_PHRASE}' was missing, appended to prompt.")


        # Generate a unique seed for variation unless specified otherwise
        seed = random.randint(0, 2**32 - 1)
        temp_output_filename = f"temp_base_{i+1}_seed{seed}.png"
        temp_output_path = os.path.join(config.TEMP_DIR, temp_output_filename)

        generated_path = image_generator.generate_image(
            pipe=sd_pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=temp_output_path,
            steps=config.SD_DEFAULT_STEPS,
            guidance_scale=config.SD_DEFAULT_GUIDANCE_SCALE,
            seed=seed
        )
        if generated_path:
            temp_image_paths.append(generated_path)
        else:
            print(f"Warning: Failed to generate base image for variation {i+1}.")

    # --- Step 3.5: Unload SD Model (Free VRAM) ---
    # Important before running text overlay if VRAM is tight
    image_generator.unload_sd_pipeline()

    # --- Step 4: Add Text Overlays ---
    print("\n=== Step 4: Adding Text Overlays ===")
    if not temp_image_paths:
        print("Error: No base images were generated successfully. Aborting text overlay.")
        cleanup_temp_files()
        return []

    for i, temp_img_path in enumerate(temp_image_paths):
        # Extract seed from filename if needed, or just use index
        base_filename = os.path.basename(temp_img_path)
        final_output_filename = f"final_thumbnail_{i+1}_{base_filename.replace('temp_base_', '')}"
        final_output_path = os.path.join(config.OUTPUT_DIR, final_output_filename)

        final_path = text_overlay.add_text_overlay(
            image_path=temp_img_path,
            user_title=user_title,
            user_concept_note=user_concept_note,
            output_path=final_output_path,
            placement_template_name=config.DEFAULT_PLACEMENT_TEMPLATE, # Use default or allow override
            font_size=config.DEFAULT_FONT_SIZE # Use default or allow override
        )
        if final_path:
            final_thumbnail_paths.append(final_path)
        else:
            print(f"Warning: Failed to add text overlay for image {i+1} ({base_filename}).")


    # --- Step 5: Cleanup ---
    print("\n=== Step 5: Cleanup ===")
    cleanup_temp_files()

    # --- Step 6: Finish ---
    end_time = time.time()
    print("\n=== Pipeline Finished ===")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    if final_thumbnail_paths:
        print(f"Successfully generated {len(final_thumbnail_paths)} thumbnails:")
        for path in final_thumbnail_paths:
            print(f"- {path}")
    else:
        print("No thumbnails were generated successfully.")

    return final_thumbnail_paths


# --- Example Usage ---
if __name__ == "__main__":
    # Get input from user or set defaults
    test_title = input("Enter Thumbnail Title: ")
    test_concept = input("Enter Concept Note: ")

    # Example input if running non-interactively:
    # test_title = "Mysteries of the Deep Ocean"
    # test_concept = "A lone submersible exploring a dark, bioluminescent trench. Encountering a huge, shadowy creature with glowing eyes. Mood should be mysterious and slightly tense. Use the kuku thumbnail style."

    if not test_title or not test_concept:
        print("Error: Title and Concept Note cannot be empty.")
    else:
        # Check API Key configuration early
        if not config.GOOGLE_API_KEY or "YOUR_GEMINI_API_KEY_HERE" in config.GOOGLE_API_KEY:
             print("CRITICAL ERROR: GOOGLE_API_KEY is not set in config.py or environment variables.")
        elif not config.DEFAULT_FONT_PATH:
             print("CRITICAL ERROR: No valid default font configured in config.py.")
        else:
             generated_files = run_thumbnail_pipeline(test_title, test_concept)
             # Optional: Display images if in a capable environment like Jupyter/Colab
             # from IPython.display import Image, display
             # for file_path in generated_files:
             #     display(Image(filename=file_path, width=300))