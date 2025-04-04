# llm_handler.py
import google.generativeai as genai
import json
import re
from config import GOOGLE_API_KEY

# Configure the Gemini client
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # Depending on severity, you might exit or let later calls fail

def clean_json_response(text):
    """Attempts to extract JSON block or list from Gemini response."""
    text = text.strip() # Remove leading/trailing whitespace

    # 1. Try finding ```json ... ``` block first
    match_block = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match_block:
        content = match_block.group(1).strip()
        # Check if the content inside the block is already a valid list or object
        if content.startswith('[') and content.endswith(']'):
            print("Cleaned: Found JSON list within ```json block.")
            return content
        if content.startswith('{') and content.endswith('}'):
             # It might be a single object OR comma-separated objects
             # Check if it looks like comma-separated objects
             if content.count('},{') > 0:
                  print("Cleaned: Found comma-separated objects within ```json block. Wrapping in list.")
                  # Wrap the comma-separated objects in brackets to make it a valid list
                  return f"[{content}]"
             else:
                  print("Cleaned: Found single JSON object within ```json block.")
                  return content # Return single object as is
        # If it's neither, maybe it's just the comma-separated objects without outer {}?
        elif content.count('},{') > 0 and content.startswith('{') and content.endswith('}'):
             print("Cleaned: Found comma-separated objects (assumed) within ```json block. Wrapping in list.")
             return f"[{content}]"


    # 2. If no ```json block, try finding content starting with [ and ending with ]
    if text.startswith('[') and text.endswith(']'):
         print("Cleaned: Found text starting/ending with []. Assuming valid JSON list.")
         return text

    # 3. Handle the specific case of comma-separated objects WITHOUT brackets
    #    Check if it starts with { , ends with }, and contains },{
    if text.startswith('{') and text.endswith('}') and text.count('},{') > 0:
        print("Cleaned: Found text starting/ending with {}. Contains ',{'. Wrapping in list.")
        return f"[{text}]" # Wrap the comma-separated objects in brackets

    # 4. Fallback: Find first '{' and last '}' (might grab too much or too little)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        potential_json = text[start:end+1].strip()
        # Check again if this substring looks like comma-separated objects
        if potential_json.startswith('{') and potential_json.endswith('}') and potential_json.count('},{') > 0:
             print("Cleaned: Found substring starting/ending with {}. Contains ',{'. Wrapping in list.")
             return f"[{potential_json}]"
        # Otherwise, assume it's a single object (less likely for this use case)
        elif potential_json.startswith('{') and potential_json.endswith('}'):
             print("Cleaned: Found substring starting/ending with {}. Assuming single object.")
             return potential_json

    # If none of the above worked
    print("Warning: Could not extract or fix JSON structure from response.")
    print(f"Original text (partially): {text[:200]}...") # Log beginning of problematic text
    return None

def call_gemini(prompt_text, expect_json=False):
    """Calls the Gemini API and optionally parses the response as JSON."""
    print("\n--- Calling Gemini API ---")
    # print(f"Prompt:\n{prompt_text}\n------------------------") # Uncomment for debugging
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21') # Or specific model like 'gemini-1.5-flash' etc
        response = model.generate_content(prompt_text)

        if not response.parts:
             print("Warning: Gemini response has no parts.")
             return None

        response_text = response.text
        # print(f"Gemini Raw Response:\n{response_text}\n-----------------------") # Debugging

        if expect_json:
            json_str = clean_json_response(response_text)
            if json_str:
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from Gemini response: {e}")
                    print(f"Problematic JSON string attempt: {json_str}")
                    return None # Failed parsing
            else:
                return None # Failed cleaning/extraction
        else:
            return response_text # Return raw text if JSON not expected

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def generate_sd_prompts_prompt(title, concept, num_variations, trigger_phrase, negative_prompt_base):
    """Creates the prompt for Gemini to generate SD prompts."""
    return f"""
    You are an expert creative assistant specializing in generating diverse prompts for an AI image generator (Stable Diffusion XL) to create compelling YouTube thumbnails.

    Objective: Generate {num_variations} distinct prompts for a thumbnail based on the provided Title and Concept Note. Each prompt must incorporate the specific style trigger phrase: "{trigger_phrase}".

    Title: "{title}"
    Concept Note: "{concept}"

    Instructions:
    1.  Analyze the Title and Concept Note to understand the core subject, mood, and implied visuals.
    2.  Generate {num_variations} different prompt variations. Each variation should explore a different aspect, composition, camera angle, color mood, or focus related to the concept.
    3.  **Crucially, EVERY prompt MUST include the phrase "{trigger_phrase}"** to activate the custom fine-tuned style.
    4.  For each variation, also generate a suitable negative prompt, starting with "{negative_prompt_base}" and adding specific exclusions relevant to the positive prompt (e.g., if prompt is 'close up', add 'wide angle' to negative). Avoid generic terms like 'bad quality' in the additions, focus on content exclusions.
    5.  Output ONLY a valid JSON list where each element is an object containing "prompt" and "negative_prompt" keys.

    Example JSON Output Format:
    [
      {{
        "prompt": "Detailed description of variation 1 including '{trigger_phrase}'...",
        "negative_prompt": "{negative_prompt_base}, excluding X, excluding Y..."
      }},
      {{
        "prompt": "Detailed description of variation 2 including '{trigger_phrase}'...",
        "negative_prompt": "{negative_prompt_base}, excluding A, excluding B..."
      }}
    ]
    """

def generate_style_suggestion_prompt(title, concept, available_fonts):
    """Creates the prompt for Gemini to suggest text styling."""
    return f"""
    You are a graphic design assistant specializing in YouTube thumbnails.
    Analyze the following title and concept note to suggest styling for the title text overlay.

    Title: "{title}"
    Concept Note: "{concept}"

    Suggest styling parameters optimized for readability and matching the concept's mood.
    Provide suggestions for:
    1. font_style_description: A brief description matching one of the available styles listed below.
    2. text_color_suggestion: A color description (e.g., 'bright white', 'light yellow', 'dark contrasting blue').
    3. text_effect: Suggest an effect for readability ('simple black outline', 'white outline', 'drop shadow', 'outer glow', 'none').

    Available font styles (choose one): {', '.join(available_fonts)}

    Output ONLY a valid JSON object with keys "font_style_description", "text_color_suggestion", "text_effect". Ensure the "font_style_description" exactly matches one of the available styles provided.
    Example:
    {{
      "font_style_description": "tech",
      "text_color_suggestion": "bright white",
      "text_effect": "simple black outline"
    }}
    """