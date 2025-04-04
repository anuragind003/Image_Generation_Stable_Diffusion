
# main_train_lora.py
import os
import sys
import time

# --- Add the current directory to sys.path to find other modules ---
# This assumes you run this script from within the lora_training directory
module_path = os.path.abspath(os.path.dirname(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)
print(f"Added {module_path} to sys.path")

# --- Import our modules ---
try:
    import config_trainer as config
    import setup_handler
    import train_runner
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure config_trainer.py, setup_handler.py, and train_runner.py exist in the same directory.")
    sys.exit(1)

def main():
    """Runs the entire LoRA training setup and launch process."""
    start_time = time.time()
    print("=== Starting LoRA Training Setup ===")

    # --- Step 1: Environment Setup ---
    print("\n--- Step 1: Environment Setup ---")
    setup_handler.mount_drive()
    setup_handler.create_training_dirs(config.INSTANCE_DATA_DIR, config.CLASS_DATA_DIR, config.OUTPUT_DIR)
    if not setup_handler.install_dependencies():
        sys.exit("Dependency installation failed. Aborting.")
    if not setup_handler.check_pytorch_cuda():
         # Warning only, might proceed on CPU if user insists, but likely impractical
        print("Warning: PyTorch CUDA check failed or CUDA not available.")
        # sys.exit("CUDA check failed. Aborting.") # Uncomment to enforce GPU

    # --- Step 2: Diffusers Repo ---
    print("\n--- Step 2: Prepare Diffusers ---")
    diffusers_repo_path = setup_handler.clone_diffusers_repo(config.DIFFUSERS_REPO_DIR)
    if not diffusers_repo_path:
        sys.exit("Failed to clone Diffusers repo. Aborting.")
    if not setup_handler.install_diffusers_from_source(diffusers_repo_path):
        sys.exit("Failed to install Diffusers from source. Aborting.")

    # --- Step 3: Build Training Command ---
    print("\n--- Step 3: Build Training Command ---")
    training_command = train_runner.build_training_command()
    if not training_command:
        sys.exit("Failed to build training command. Aborting.")

    # --- Step 4: Run Training ---
    print("\n--- Step 4: Execute Training ---")
    # The command needs to be run from the script's directory
    success = train_runner.run_training(training_command, config.TRAIN_SCRIPT_DIR)

    # --- Finish ---
    end_time = time.time()
    print("\n=== LoRA Training Pipeline Finished ===")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    if success:
        print("Training completed. Check outputs.")
    else:
        print("Training failed. Check logs.")

if __name__ == "__main__":
    # --- Change to the script's directory BEFORE running main ---
    # This ensures relative imports and config loading work correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Changing working directory to: {script_dir}")
    os.chdir(script_dir)
    main()