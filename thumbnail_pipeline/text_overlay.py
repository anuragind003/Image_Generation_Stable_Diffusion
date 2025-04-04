# text_overlay.py
import os
from PIL import Image, ImageDraw, ImageFont
import llm_handler # Import the module to use its functions
from config import (
    FONT_MAPPING,
    DEFAULT_FONT_PATH,
    DEFAULT_TEXT_COLOR,
    DEFAULT_OUTLINE_COLOR,
    DEFAULT_OUTLINE_WIDTH,
    DEFAULT_FONT_SIZE,
    DEFAULT_PLACEMENT_TEMPLATE
)

# --- Placement Templates ---
def get_placement_bottom_center(img_width, img_height, text_width, text_height, padding=20):
    x = (img_width - text_width) / 2
    y = img_height - text_height - padding
    return int(x), int(y)

def get_placement_bottom_left(img_width, img_height, text_width, text_height, padding=20):
    x = padding
    y = img_height - text_height - padding
    return int(x), int(y)

PLACEMENT_TEMPLATES = {
    "bottom_center": get_placement_bottom_center,
    "bottom_left": get_placement_bottom_left,
    # Add more placement functions here if needed
}

# --- Color Translation (Basic) ---
def translate_color(color_suggestion_str, default_color):
    color_suggestion_str = color_suggestion_str.lower()
    # Prioritize specific color names if found
    if "white" in color_suggestion_str: return (255, 255, 255)
    if "black" in color_suggestion_str: return (0, 0, 0)
    if "yellow" in color_suggestion_str: return (255, 255, 100)
    if "red" in color_suggestion_str: return (255, 50, 50)
    if "blue" in color_suggestion_str: return (100, 100, 255)
    if "green" in color_suggestion_str: return (50, 200, 50)
    if "orange" in color_suggestion_str: return (255, 165, 0)
    if "purple" in color_suggestion_str: return (128, 0, 128)
    if "pink" in color_suggestion_str: return (255, 192, 203)
    # Add more - consider a color mapping dictionary for better control
    print(f"Color '{color_suggestion_str}' not recognized, using default.")
    return default_color


# --- Get Style Suggestions (Uses llm_handler) ---
def get_style_suggestions(user_title, user_concept_note):
    """Asks Gemini for text styling suggestions via llm_handler."""
    available_fonts = list(FONT_MAPPING.keys())
    prompt = llm_handler.generate_style_suggestion_prompt(user_title, user_concept_note, available_fonts)
    suggestions = llm_handler.call_gemini(prompt, expect_json=True)

    # Validate suggestions
    if suggestions and isinstance(suggestions, dict):
        if not all(k in suggestions for k in ["font_style_description", "text_color_suggestion", "text_effect"]):
             print("Warning: Gemini style response missing keys. Using defaults.")
             return None
        # Ensure suggested font is actually available
        font_key = suggestions["font_style_description"].lower()
        if font_key not in FONT_MAPPING or not os.path.exists(FONT_MAPPING[font_key]):
             print(f"Warning: Suggested font style '{font_key}' not available or path invalid. Using default.")
             return None # Force defaults if font is unusable

        print(f"Received Gemini Styling Suggestions: {suggestions}")
        return suggestions
    else:
        print("Failed to get valid style suggestions from Gemini. Using default styling.")
        return None


# --- Add Text Overlay (Main Function) ---
def add_text_overlay(
    image_path,
    user_title,
    user_concept_note,
    output_path,
    placement_template_name=DEFAULT_PLACEMENT_TEMPLATE,
    font_size=DEFAULT_FONT_SIZE,
    ):
    """Adds text overlay using Gemini suggestions and template placement."""
    print(f"\n--- Adding Text Overlay to {os.path.basename(image_path)} ---")

    # Check if default font path is valid (critical)
    if DEFAULT_FONT_PATH is None:
        print("Error: Cannot proceed with text overlay due to invalid default font configuration.")
        return None

    # --- Step C: Get Styling Suggestions ---
    suggestions = get_style_suggestions(user_title, user_concept_note)

    # --- Determine final styling parameters ---
    if suggestions:
        font_style_key = suggestions.get("font_style_description", list(FONT_MAPPING.keys())[0]).lower()
        # Use the validated path from FONT_MAPPING
        font_path = FONT_MAPPING.get(font_style_key, DEFAULT_FONT_PATH)
        text_color = translate_color(suggestions.get("text_color_suggestion", "white"), DEFAULT_TEXT_COLOR)
        effect = suggestions.get("text_effect", "simple black outline").lower()
    else:
        # Use defaults if Gemini failed or suggestions invalid
        font_path = DEFAULT_FONT_PATH
        text_color = DEFAULT_TEXT_COLOR
        effect = "simple black outline"

    # Determine outline based on effect
    apply_outline = False
    outline_color = DEFAULT_OUTLINE_COLOR
    outline_width = DEFAULT_OUTLINE_WIDTH
    # Add more sophisticated effect logic if needed
    if "outline" in effect:
        apply_outline = True
        outline_color = translate_color(effect.replace(" outline", ""), DEFAULT_OUTLINE_COLOR) # Try to parse color from effect name
    elif "drop shadow" in effect:
         apply_outline = True
         outline_color = (0, 0, 0, 150) # Semi-transparent black shadow
         outline_width = DEFAULT_OUTLINE_WIDTH + 2

    # --- Step B & E: Render Text using Pillow with Template Placement ---
    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, font_size)

        # Calculate text size using textbbox
        text_bbox = draw.textbbox((0, 0), user_title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Get placement function and coordinates
        placement_function = PLACEMENT_TEMPLATES.get(placement_template_name, PLACEMENT_TEMPLATES[DEFAULT_PLACEMENT_TEMPLATE])
        x, y = placement_function(img.width, img.height, text_width, text_height)

        # Apply outline/effect
        if apply_outline:
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                     is_within_distance = (dx * dx + dy * dy) <= (outline_width * outline_width + 1)
                     if is_within_distance:
                         draw.text((x + dx, y + dy), user_title, font=font, fill=outline_color)

        # Draw main text
        draw.text((x, y), user_title, font=font, fill=text_color)

        # Save final image
        if output_path.lower().endswith(".jpg") or output_path.lower().endswith(".jpeg"):
             img = img.convert("RGB")
        img.save(output_path)
        print(f"Text overlay complete. Final image saved to: {output_path}")
        return output_path

    except FileNotFoundError:
         print(f"CRITICAL Error: Font file not found at '{font_path}'. Check config.py and ensure fonts exist.")
         return None
    except Exception as e:
        print(f"Error during text overlay for {image_path}: {e}")
        return None
    finally:
         if 'img' in locals():
             img.close()