
# train_runner.py
import os
import subprocess
import sys
# Import the config directly
import config_trainer as config

def build_training_command():
    """Builds the accelerate launch command string from config."""
    print("Building training command...")

    # --- VAE ---
    vae_path_arg = ""
    if hasattr(config, 'PRETRAINED_VAE_NAME_OR_PATH') and config.PRETRAINED_VAE_NAME_OR_PATH:
        vae_path_arg = f"--pretrained_vae_model_name_or_path='{config.PRETRAINED_VAE_NAME_OR_PATH}'"

    # --- Boolean Flags ---
    use_8bit_adam_flag = "--use_8bit_adam" if config.USE_8BIT_ADAM else ""
    gradient_checkpointing_flag = "--gradient_checkpointing" if config.GRADIENT_CHECKPOINTING else ""
    enable_xformers_flag = "--enable_xformers_memory_efficient_attention" if config.ENABLE_XFORMERS else ""

    # --- Prior Preservation ---
    prior_preservation_args = ""
    num_class_images = 0
    if config.WITH_PRIOR_PRESERVATION:
        if os.path.exists(config.CLASS_DATA_DIR) and os.path.isdir(config.CLASS_DATA_DIR):
            class_files = [f for f in os.listdir(config.CLASS_DATA_DIR) if os.path.isfile(os.path.join(config.CLASS_DATA_DIR, f))]
            num_class_images = len(class_files)

        if num_class_images > 0:
            print(f"Prior Preservation enabled. Found {num_class_images} class images in {config.CLASS_DATA_DIR}.")
            prior_preservation_args = (
                f"--with_prior_preservation "
                f"--prior_loss_weight={config.PRIOR_LOSS_WEIGHT} "
                f"--class_data_dir='{config.CLASS_DATA_DIR}' "
                f"--class_prompt='{config.CLASS_PROMPT}' "
                f"--num_class_images={num_class_images}" # Let accelerate handle sampling based on count
            )
        else:
            print(f"Warning: Prior Preservation was enabled in config, but no class images found in '{config.CLASS_DATA_DIR}'. Disabling.")
    else:
        print("Prior Preservation disabled.")


    # --- Build Command List ---
    cmd_parts = [
        "accelerate", "launch", config.TRAIN_SCRIPT_PATH,
        # --- Model and Data ---
        f"--pretrained_model_name_or_path='{config.PRETRAINED_MODEL_NAME_OR_PATH}'",
        vae_path_arg,
        f"--instance_data_dir='{config.INSTANCE_DATA_DIR}'",
        f"--output_dir='{config.OUTPUT_DIR}'",
        # --- Prompts ---
        f"--instance_prompt='{config.INSTANCE_PROMPT}'",
        # --- Core Training Params ---
        f"--resolution={config.RESOLUTION}",
        f"--train_batch_size={config.TRAIN_BATCH_SIZE}",
        f"--gradient_accumulation_steps={config.GRADIENT_ACCUMULATION_STEPS}",
        f"--max_train_steps={config.MAX_TRAIN_STEPS}",
        # --- Learning Rate ---
        f"--learning_rate={config.LEARNING_RATE}",
        f"--lr_scheduler='{config.LR_SCHEDULER}'",
        f"--lr_warmup_steps={config.LR_WARMUP_STEPS}",
        # --- LoRA Specific ---
        f"--rank={config.LORA_RANK}",
        # --- Validation & Saving ---
        f"--validation_prompt='{config.VALIDATION_PROMPT}'",
        f"--validation_epochs={config.VALIDATION_EPOCHS}",
        f"--checkpointing_steps={config.SAVE_STEPS}",
        # --- Optimizations & Precision ---
        f"--mixed_precision='{config.MIXED_PRECISION}'",
        use_8bit_adam_flag,
        gradient_checkpointing_flag,
        enable_xformers_flag,
        # --- Other ---
        f"--seed={config.SEED}",
        f"--report_to='{config.REPORT_TO}'",
        # --- Conditional Prior Preservation ---
        prior_preservation_args
    ]

    # Filter empty strings and construct command
    command = " ".join(part for part in cmd_parts if part)
    print("Training command built successfully.")
    # print(f"Command: {command}") # Uncomment for debugging
    return command


def run_training(command, working_directory):
    """Executes the training command in the specified directory."""
    if not command:
        print("Error: Training command is empty.")
        return False

    print(f"\n--- Starting LoRA Training ---")
    print(f"Working Directory: {working_directory}")
    print(f"Executing command: {command}")
    print("-" * 30)

    original_dir = os.getcwd()
    try:
        # Change to the directory where the script expects to be run
        if not os.path.exists(working_directory):
             print(f"Error: Working directory '{working_directory}' does not exist.")
             return False
        os.chdir(working_directory)
        print(f"Changed directory to: {os.getcwd()}")

        # Execute the command using subprocess
        # Using sys.executable ensures we use the same python env
        # Splitting the command string appropriately for subprocess might be needed
        # For `accelerate launch`, it's often easier to run via shell=True,
        # but be mindful of security implications if command was user-provided.
        # Here, we construct the command, so it's relatively safe.
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        # Stream output
        if process.stdout:
             for line in iter(process.stdout.readline, ''):
                 print(line, end='') # Print line by line

        process.wait() # Wait for completion
        return_code = process.returncode

        os.chdir(original_dir) # Change back to original directory
        print("-" * 30)

        if return_code == 0:
            print("Training process completed successfully.")
            print(f"Check LoRA outputs in: {config.OUTPUT_DIR}")
            return True
        else:
            print(f"Training process failed with exit code: {return_code}")
            print("Review the output logs above for errors.")
            print("Common issues: Out-of-memory (OOM), incorrect paths, missing files, config errors.")
            return False

    except FileNotFoundError:
        print(f"Error: 'accelerate' command not found. Ensure 'accelerate' is installed and in PATH.")
        os.chdir(original_dir)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during training execution: {e}")
        os.chdir(original_dir)
        return False