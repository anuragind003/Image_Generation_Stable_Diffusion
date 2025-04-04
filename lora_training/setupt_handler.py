
# setup_handler.py
import os
import subprocess
import sys
from google.colab import drive

def mount_drive(mount_point="/content/drive"):
    """Mounts Google Drive if not already mounted."""
    if not os.path.exists(mount_point) or not os.listdir(mount_point):
        print(f"Mounting Google Drive at {mount_point}...")
        try:
            drive.mount(mount_point, force_remount=True) # Force remount can help if stale
            print("Google Drive mounted successfully.")
        except Exception as e:
            print(f"Error mounting Google Drive: {e}")
            sys.exit("Stopping: Google Drive mount failed.")
    else:
        print("Google Drive already mounted.")

def create_training_dirs(instance_dir, class_dir, output_dir):
    """Creates necessary directories for training data and output."""
    print("Ensuring training directories exist...")
    os.makedirs(instance_dir, exist_ok=True)
    os.makedirs(class_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"- Instance Image Dir: {instance_dir}")
    print(f"- Class Image Dir (Optional): {class_dir}")
    print(f"- LoRA Output Dir: {output_dir}")
    print("\n>>> ACTION REQUIRED (Before Training): <<<\")")
    print(f">>> Upload your INSTANCE images (.jpg/.png) to: '{instance_dir}'")
    print(f">>> If using Prior Preservation, upload CLASS images to: '{class_dir}'")
    print("="*40)

def install_dependencies():
    """Installs necessary Python packages."""
    print("Installing dependencies...")
    try:
        # Install PyTorch matching Colab CUDA (usually 11.8 or 12.x) - check current Colab version if needed
        # Using --force-reinstall can help ensure correct versions
        print("Installing PyTorch & Torchvision...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-cache-dir",
                        "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"], check=True) # Adjust cu118 if needed

        print("Installing other libraries (transformers, accelerate, etc.)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "transformers", "accelerate", "bitsandbytes", "ftfy", "gradio",
                        "safetensors", "tqdm", "pillow", "dotenv"], check=True) # Added dotenv, pillow, tqdm

        # Optionally install xformers - can fail sometimes
        try:
            print("Attempting to install xformers...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xformers"], check=True)
            print("xformers installed successfully.")
        except subprocess.CalledProcessError:
            print("Warning: xformers installation failed. Training will proceed without it if ENABLE_XFORMERS=False in config.")

        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during dependency installation: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during installation: {e}")
        return False

def check_pytorch_cuda():
    """Verifies PyTorch and CUDA setup (formerly Cell 2.5)."""
    print("\n--- Verifying PyTorch/CUDA Setup ---")
    try:
        import torch
        import torchvision
        print(f"PyTorch version: {torch.__version__}")
        print(f"Torchvision version: {torchvision.__version__}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA available: True")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            print(f"PyTorch CUDA available: False - Training will be extremely slow or fail.")
            return False # Indicate failure if no CUDA

        # Basic torchvision check
        # torchvision.extension is deprecated, use ops instead
        print(f"Torchvision Ops available: {torchvision.ops.is_available()}")
        print("Torchvision seems compatible.")
        print("-" * 30)
        return True
    except ImportError as e:
        print(f"Error importing torch/torchvision: {e}. Installation likely failed.")
        print("-" * 30)
        return False
    except Exception as e:
        print(f"Error during PyTorch/CUDA check: {e}")
        print("-" * 30)
        return False


def clone_diffusers_repo(target_dir="/content/diffusers"):
    """Clones the Hugging Face Diffusers repository."""
    if not os.path.exists(target_dir):
        print(f"Cloning Diffusers repository into {target_dir}...")
        try:
            # Use subprocess for better error handling than !git clone
            subprocess.run(["git", "clone", "https://github.com/huggingface/diffusers.git", target_dir], check=True)
            print("Diffusers repository cloned successfully.")
            return target_dir
        except subprocess.CalledProcessError as e:
            print(f"Error cloning Diffusers repository: {e}")
            return None
        except FileNotFoundError:
             print("Error: 'git' command not found. Make sure git is installed in your Colab environment.")
             return None
    else:
        print(f"Diffusers repository already exists at {target_dir}.")
        # Optional: Add git pull logic here if you want to update
        # try:
        #     print("Pulling latest changes...")
        #     subprocess.run(["git", "-C", target_dir, "pull"], check=True)
        # except Exception as e:
        #     print(f"Warning: Failed to pull latest changes for diffusers repo: {e}")
        return target_dir

def install_diffusers_from_source(repo_path):
    """Installs the diffusers library from the cloned source code."""
    if not repo_path or not os.path.exists(repo_path):
        print("Error: Diffusers repository path not valid.")
        return False
    print(f"Installing diffusers from source directory: {repo_path}...")
    original_dir = os.getcwd()
    try:
        os.chdir(repo_path) # Navigate into the repo
        # Use -e for editable install
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
        print("Diffusers installed from source successfully.")
        os.chdir(original_dir) # Go back to original directory
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing diffusers from source: {e}")
        os.chdir(original_dir)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during diffusers source install: {e}")
        os.chdir(original_dir)
        return False