
# AI Thumbnail Generation Pipeline with Stable Diffusion LoRA

This project provides a pipeline to fine-tune Stable Diffusion (specifically SDXL using LoRA) for a particular visual style and then use that fine-tuned model, along with Google's Gemini LLM, to generate YouTube-style thumbnails based on a title and concept note.

**Features:**

*   **LoRA Fine-tuning:** Scripts to train a LoRA adapter on SDXL using your own style images (via `lora_training/`).
*   **AI Prompt Generation:** Uses Google Gemini to convert a high-level concept note and title into specific, varied prompts suitable for Stable Diffusion.
*   **AI Text Styling:** Uses Google Gemini to suggest appropriate font styles, colors, and effects for the title overlay based on the concept.
*   **Image Generation:** Generates multiple thumbnail base images using the fine-tuned LoRA style.
*   **Automated Text Overlay:** Renders the provided title onto the generated images using Pillow, applying AI-suggested styles and template-based placement.
*   **Modular Structure:** Code is organized into separate modules for training and generation pipelines.

## Project Structure

├── db_lora_sdxl_project/       # Main project directory
│   ├── instance_images/        # Style training images (local/Drive)
│   ├── class_images/           # Optional regularization images (local/Drive)
│   ├── output_lora/            # Trained LoRA model output (local/Drive)
│
│   ├── thumbnail_pipeline/     # Thumbnail Generation Code
│   │   ├── config.py           # Configuration file for generation (API keys, paths, LoRA details)
│   │   ├── llm_handler.py      # Gemini API interactions
│   │   ├── image_generator.py  # Stable Diffusion image generation logic
│   │   ├── text_overlay.py     # Text rendering logic
│   │   ├── main_pipeline.py    # Main script to run the thumbnail generation
│   │   ├── fonts/              # Font files directory (Required .ttf/.otf files)
│   │   │   └── DejaVuSans-Bold.ttf  # Example font
│   │   └── requirements.txt    # Dependencies for the thumbnail generation pipeline
│
│   ├── lora_training/          # LoRA Model Training Code
│   │   ├── config_trainer.py   # Configuration file for training (paths, hyperparameters)
│   │   ├── setup_handler.py    # Installs required dependencies and clones the diffusers repository
│   │   ├── train_runner.py     # Builds and runs the training command
│   │   └── main_train_lora.py  # Main script to execute LoRA training
│
│   ├── .env                    # Optional: Stores API keys securely (add to .gitignore!)
│   ├── .gitignore              # Specifies intentionally untracked files by Git
│   ├── LICENSE                 # Project license (e.g., MIT)
│   ├── README.md               # This file (project documentation)
│   └── requirements.txt        # Combined Python dependencies for the entire project

## Setup

**Prerequisites:**

*   Git
*   Python 3.9+
*   Google Drive account (if running on Colab and following current path structure)
*   GPU with sufficient VRAM (especially for SDXL training/inference) - ideally CUDA-enabled.
*   Google AI Studio API Key (for Gemini)

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/anuragind003/Image_Generation_Stable_Diffusion.git
    cd Image_Generation_Stable_Diffusion
    ```

2.  **Set Up Google Drive Structure (Manual):**
    *   Create the `db_lora_sdxl_project` folder on your Google Drive (or locally if adapting paths).
    *   Inside `db_lora_sdxl_project`, create `instance_images`, `class_images`, and `output_lora`.
    

3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

4.  **Install Dependencies:**
    *   Install PyTorch separately first, matching your CUDA version (refer to [PyTorch website](https://pytorch.org/get-started/locally/)). Example for CUDA 11.8:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   Install project dependencies from the *root* `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```

5.  **API Keys (CRITICAL):**
    *   Obtain your Google AI Studio API Key.
    *   **Option A (Recommended):** Create a file named `.env` in the project root directory (where `README.md` is). Add your key like this:
        ```dotenv
        # .env
        GOOGLE_API_KEY="AIzaSyB...your...actual...key...string..."
        ```
        *(Ensure `.env` is listed in your `.gitignore` file!)*
    *   **Option B (Less Secure):** Directly edit `thumbnail_pipeline/config.py` and replace the placeholder in the `GOOGLE_API_KEY` line. Do the same for `lora_training/config_trainer.py` if needed (though it doesn't use Gemini). Avoid committing keys directly to GitHub.

6.  **Add Fonts:**
    *   Place the `.ttf` or `.otf` font files you want to use for the title overlay inside the `thumbnail_pipeline/fonts/` directory.
    *   Update the `FONT_MAPPING` dictionary in `thumbnail_pipeline/config.py` to correctly list your available fonts and their filenames.

7.  **Place Pre-trained LoRA (for Generation):**
    *   If you have already trained a LoRA model, copy the `.safetensors` file into the location specified by `LORA_FILE_PATH` in `thumbnail_pipeline/config.py` (e.g., `/content/drive/MyDrive/db_lora_sdxl_project/output_lora/pytorch_lora_weights.safetensors`).

## Usage

**1. Training a Custom LoRA Model:**

*   **Prepare Data:** Upload your style instance images to `db_lora_sdxl_project/instance_images/`. If using prior preservation, upload class images to `db_lora_sdxl_project/class_images/`.
*   **Configure:** Edit `lora_training/config_trainer.py` to set:
    *   Paths (`DRIVE_PROJECT_BASE`, etc.)
    *   `UNIQUE_TOKEN` and `CLASS_NAME` for your style.
    *   Training hyperparameters (`MAX_TRAIN_STEPS`, `LEARNING_RATE`, etc.).
    *   `WITH_PRIOR_PRESERVATION` (True/False).
*   **Run Training (from project root directory):**
    ```bash
    python lora_training/main_train_lora.py
    ```
    *(Note: On Colab, you might run `!python /content/drive/MyDrive/lora_training/main_train_lora.py` after mounting Drive).*
*   **Output:** The trained LoRA (`.safetensors`) will be saved in `db_lora_sdxl_project/output_lora/`.

**2. Generating Thumbnails:**

*   **Configure:** Edit `thumbnail_pipeline/config.py` to set:
    *   `GOOGLE_API_KEY` (if not using `.env`).
    *   `LORA_FILE_PATH` (point to your trained `.safetensors` file).
    *   `LORA_TRIGGER_PHRASE` (must match the trigger used during training).
    *   Verify `FONT_DIR` and `FONT_MAPPING`.
    *   Adjust `OUTPUT_DIR` if desired.
*   **Run Generation (from project root directory):**
    ```bash
    python thumbnail_pipeline/main_pipeline.py
    ```
    *(Note: On Colab, you might run `!python /content/drive/MyDrive/thumbnail_pipeline/main_pipeline.py` after mounting Drive and changing directory).*
*   **Input:** The script will prompt you to enter a `Title` and a `Concept Note`.
*   **Output:** Final thumbnail images (with text overlays) will be saved in the directory specified by `OUTPUT_DIR` in `thumbnail_pipeline/config.py`.

## Configuration Files

*   `lora_training/config_trainer.py`: Controls all parameters for the LoRA training process.
*   `thumbnail_pipeline/config.py`: Controls parameters for the thumbnail generation process, including API keys, model paths, LoRA details, and text overlay settings.

## Notes

*   This project is designed with Google Colab in mind, particularly regarding path structures involving Google Drive and dependencies like PyTorch+CUDA. Adapting for local use may require path modifications and ensuring correct local environment setup.
*   Running SDXL models requires significant computational resources (GPU VRAM). Training is particularly demanding.
*   Using the Google Gemini API incurs costs based on usage. Refer to Google AI Platform pricing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
