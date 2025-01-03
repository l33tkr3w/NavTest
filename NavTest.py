import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
import pyautogui
import requests
import json
import base64
from time import sleep
from pywinauto import Desktop
import subprocess  # To launch Google Chrome
import os  # For environment detection
import logging
import sys  # For platform detection
import re  # For regex operations
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='automation_debug.log',  # Log file name
    filemode='a',                      # Append mode
    level=logging.DEBUG,               # Set to DEBUG to capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# PyAutoGUI failsafe
pyautogui.FAILSAFE = True

# Device configuration 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load PTA-1 model and processor
try:
    pta_processor = AutoProcessor.from_pretrained("AskUI/PTA-1", trust_remote_code=True)
    pta_model = AutoModelForCausalLM.from_pretrained(
        "AskUI/PTA-1",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    logging.info("PTA-1 model and processor loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load PTA-1 model or processor: {e}")
    print(f"Error: Failed to load PTA-1 model or processor: {e}")

# API Endpoints
QWEN_API_URL = "http://localhost:11434/api/generate"
MINICPM_API_URL = "http://localhost:11434/api/generate"
HEADERS = {"Content-Type": "application/json"}

# Global browsing state
isBrowsing = False  # Indicates whether the model is in browsing mode

def image_to_base64(image_path):
    """Convert an image to a base64-encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        logging.debug(f"Converted image '{image_path}' to base64.")
        return base64_image
    except Exception as e:
        logging.error(f"Failed to convert image to base64: {e}")
        print(f"Error: Failed to convert image to base64: {e}")
        return None

def start_browsing_mode(url):
    """Start browsing mode and open the specified URL in Google Chrome."""
    global isBrowsing
    if isBrowsing:
        logging.info("Browsing mode is already active. Skipping 'start_browsing_mode' action.")
        print("Browsing mode is already active. Skipping 'start_browsing_mode' action.")
        return False  # Indicate that browsing mode was already active

    isBrowsing = True

    try:
        # Open Google Chrome to the specified URL
        if os.name == "nt":  # Windows
            subprocess.Popen(["start", "chrome", url], shell=True)
        elif sys.platform == "darwin":  # macOS
            subprocess.Popen(["open", "-a", "Google Chrome", url])
        else:  # For Linux or other OS, adjust as needed
            subprocess.Popen(["google-chrome", url])

        logging.info(f"Browsing mode started. Opened URL: {url}")
        print(f"Browsing mode started. Opened URL: {url}")
        # Increase delay after starting the browser to allow the page to fully load
        sleep(4)  # Wait a few seconds to allow the browser to load completely
        return True  # Indicate successful execution
    except Exception as e:
        logging.error(f"Failed to start browsing mode: {e}")
        print(f"Error: Failed to start browsing mode: {e}")
        return False  # Indicate failure

def stop_browsing_mode():
    """Stop browsing mode."""
    global isBrowsing
    if not isBrowsing:
        logging.info("Browsing mode is already inactive. Skipping 'stop_browsing_mode' action.")
        print("Browsing mode is already inactive. Skipping 'stop_browsing_mode' action.")
        return False  # Indicate that browsing mode was already inactive

    isBrowsing = False
    logging.info("Browsing mode ended.")
    print("Browsing mode ended.")
    return True  # Indicate successful execution

def format_completed_steps(completed_steps, use_vision):
    """Format the completed steps section regardless of vision mode."""
    if not completed_steps:
        return ""
    
    steps = "\n".join([
        f"- Action: {step['action']}, Element: {step['element_description']}, "
        f"Target: {step.get('target_description', '')}, Text: {step.get('text', '')}"
        for step in completed_steps
    ])
    return f"The following steps have already been completed:\n{steps}\n"

def format_skipped_steps(skipped_steps, use_vision):
    """Format the skipped steps section regardless of vision mode."""
    if not skipped_steps:
        return ""
    
    steps = "\n".join([
        f"- Action: {step['action']}, Element: {step['element_description']}, "
        f"Target: {step.get('target_description', '')}, Text: {step.get('text', '')}"
        for step in skipped_steps
    ])
    return f"The following steps were suggested but skipped (already completed):\n{steps}\n"

def update_breadcrumbs(context, step):
    """
    Update the breadcrumbs trail with the current step.
    This helps the AI keep track of its progress and recover from errors.
    """
    if 'breadcrumbs' not in context:
        context['breadcrumbs'] = []

    # Add the current step to the breadcrumbs trail
    context['breadcrumbs'].append({
        'action': step['action'],
        'element_description': step.get('element_description', ''),
        'target_description': step.get('target_description', ''),
        'text': step.get('text', ''),
        'timestamp': datetime.now().isoformat()  # Add a timestamp for tracking
    })

    # Limit the breadcrumbs trail to the last 10 steps to avoid memory bloat
    if len(context['breadcrumbs']) > 10:
        context['breadcrumbs'].pop(0)

def get_breadcrumbs_summary(context):
    """
    Generate a summary of the breadcrumbs trail for the AI to understand its progress.
    """
    if 'breadcrumbs' not in context or not context['breadcrumbs']:
        return "No recent steps recorded."

    summary = "Recent steps (breadcrumbs):\n"
    for i, step in enumerate(context['breadcrumbs'], 1):
        summary += (
            f"{i}. Action: {step['action']}, "
            f"Element: {step['element_description']}, "
            f"Target: {step.get('target_description', '')}, "
            f"Text: {step.get('text', '')}, "
            f"Time: {step['timestamp']}\n"
        )
    return summary

def prepare_prompt(context):
    """
    Prepare the prompt for the Language Model based on the current context.
    Modified to include the breadcrumbs summary.
    """
    completed_steps = context.get('completed_steps', [])
    skipped_steps = context.get('skipped_steps', [])
    objective = context.get('objective', '')
    
    # Always show completed steps regardless of vision mode
    completed_steps_text = format_completed_steps(completed_steps, True)
    skipped_steps_text = format_skipped_steps(skipped_steps, True)

    # Analyze current form state
    form_state = {
        'current_page': None,
        'filled_fields': {},
        'current_form': None,
        'last_action': None,
        'required_fields': set()
    }
    
    # Track state from completed steps
    for step in completed_steps:
        action = step['action'].lower()
        desc = step.get('element_description', '').lower()
        text = step.get('text', '')

        if action == 'start_browsing_mode':
            form_state['current_page'] = 'initial'
            if 'reddit' in text.lower():
                form_state['current_page'] = 'reddit_main'
                form_state['required_fields'] = {'title', 'body'}

        if 'create post' in desc:
            form_state['current_page'] = 'create_post'
            form_state['current_form'] = 'reddit_post'
            
        if action == 'type':
            if 'title' in desc.lower():
                form_state['filled_fields']['title'] = text
            elif 'body' in desc.lower():
                form_state['filled_fields']['body'] = text

        form_state['last_action'] = {
            'action': action,
            'description': desc,
            'text': text
        }

    # Format state context text
    state_lines = []
    if form_state['current_page']:
        state_lines.append(f"Current page: {form_state['current_page']}")
    if form_state['current_form']:
        state_lines.append(f"Current form: {form_state['current_form']}")
    if form_state['filled_fields']:
        state_lines.append("Completed fields:")
        for field, value in form_state['filled_fields'].items():
            state_lines.append(f"- {field}: filled")
    if form_state['required_fields']:
        missing = form_state['required_fields'] - set(form_state['filled_fields'].keys())
        if missing:
            state_lines.append("Missing required fields:")
            for field in missing:
                state_lines.append(f"- {field}")

    state_text = "\n".join(state_lines)

    # Add breadcrumbs summary to the prompt
    breadcrumbs_summary = get_breadcrumbs_summary(context)

    browsing_state = "ACTIVE" if isBrowsing else "INACTIVE"

    browsing_instructions = (
        "- To browse the internet, use 'start_browsing_mode' with a URL. Example:\n"
        "  {\n"
        "    \"action\": \"start_browsing_mode\",\n"
        "    \"element_description\": \"\",\n"
        "    \"target_description\": \"\",\n"
        "    \"text\": \"https://www.example.com\"\n"
        "  }\n"
        "- The 'text' field must contain a valid URL.\n"
        "- If browsing mode is active, do not start it again. Use mouse and keyboard actions.\n"
        "- When finished, use 'stop_browsing_mode'.\n"
        "- After right-clicking, you can click on context menu items like 'Open image in new window'.\n"
        "- Context menus stay open until you make a selection or click elsewhere.\n"
        "- For right-click operations, follow this sequence:\n"
        "  1. Right-click on the target element\n"
        "  2. Click on the desired menu option\n"
    )

    allowed_actions = [
        "click",
        "doubleclick",
        "rightclick",
        "drag",
        "hover",
        "move",
        "type",
        "start_browsing_mode",
        "stop_browsing_mode"
    ]

    strict_instructions = (
        "IMPORTANT:\n"
        "- DO NOT use Markdown formatting, code fences, or additional explanations.\n"
        "- ONLY output a single JSON object.\n"
        "- If no further action is needed (objective is completed), output:\n"
        "  {\n"
        "    \"action\": \"no_action\",\n"
        "    \"element_description\": \"\",\n"
        "    \"target_description\": \"\",\n"
        "    \"text\": \"\"\n"
        "  }\n"
        "- No other text should be outside the JSON object.\n"
        "- Avoid repeating any actions that have already been completed (see completed steps list).\n"
        "- Do not repeat field entries that are already filled.\n"
        "- Complete all required fields before submitting forms.\n"
        "- Do not use any special characters or formatting (like **, ##, etc.) in text fields."
    )

    return (
        f"You are an intelligent assistant responsible for guiding a Windows 11 application in completing tasks.\n"
        f"The application allows interaction with UI elements by providing descriptions and performing actions.\n"
        f"Your role is to carefully reason about and provide the next atomic step needed to achieve the objective.\n\n"
        f"{completed_steps_text}"
        f"{skipped_steps_text}"
        f"Objective: {objective}\n\n"
        f"CURRENT STATE:\n{state_text}\n\n"
        f"Browsing State: {browsing_state}\n\n"
        f"{breadcrumbs_summary}\n\n"  # Add breadcrumbs summary here
        f"{browsing_instructions}\n\n"
        "Instructions:\n"
        f"- Provide the next atomic step using only these actions: {', '.join([f'\'{action.capitalize()}\'' for action in allowed_actions])}.\n"
        "- Each step must include:\n"
        "  - \"action\": (e.g. 'click', 'type')\n"
        "  - \"element_description\": The element to target.\n"
        "  - \"target_description\" (if applicable for 'drag').\n"
        "  - \"text\" (if applicable) for typing or browsing.\n\n"
        "Output format (JSON only):\n"
        "{\n"
        "  \"action\": \"<action>\",\n"
        "  \"element_description\": \"<element_description>\",\n"
        "  \"target_description\": \"<target_description if applicable>\",\n"
        "  \"text\": \"<text if applicable>\"\n"
        "}\n\n"
        f"{strict_instructions}"
    )

def get_next_step(context, retry_limit=5):
    allowed_actions = [
        "click",
        "doubleclick",
        "rightclick",
        "drag",
        "hover",
        "move",
        "type",
        "start_browsing_mode",
        "stop_browsing_mode",
        "no_action"
    ]

    for attempt in range(1, retry_limit + 1):
        prompt = prepare_prompt(context)
        payload = {
            "model": "qwen2.5-coder:14b",
            "prompt": prompt,
            "num_ctx": 16000
        }

        try:
            response = requests.post(QWEN_API_URL, json=payload, headers=HEADERS, stream=True)
            response.raise_for_status()

            response_fragments = []
            done_encountered = False

            # Process each line as a separate JSON object
            for line in response.iter_lines():
                if line:
                    fragment_line = line.decode("utf-8", errors="replace").strip()
                    try:
                        fragment_json = json.loads(fragment_line)
                        if "response" in fragment_json:
                            response_fragments.append(fragment_json["response"])
                        if fragment_json.get("done") is True:
                            done_encountered = True
                            break
                    except json.JSONDecodeError:
                        logging.warning(f"Skipped non-JSON line: {fragment_line}")
                        print(f"Skipped non-JSON line: {fragment_line}")

            full_response = "".join(response_fragments).strip()
            if not full_response:
                logging.warning("Empty response from LLM after assembling fragments.")
                print("Warning: Empty response from LLM after assembling fragments.")
                continue

            cleaned = full_response.replace("```json", "").replace("```", "").strip()

            step = None
            try:
                # Attempt direct JSON parse
                step = json.loads(cleaned)
            except json.JSONDecodeError:
                # If direct parsing fails, try extracting a JSON object
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1:
                    json_str = cleaned[start:end+1].strip()
                    try:
                        step = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse extracted JSON: {e}")
                        print(f"Error: Failed to parse extracted JSON: {e}")
                else:
                    logging.warning("No braces found in response to extract JSON.")
                    print("Warning: No braces found in response to extract JSON.")

            if not step:
                logging.warning("No valid JSON object found in Qwen response. Retrying...")
                print("Warning: No valid JSON object found in Qwen response. Retrying...")
                continue

            action_lower = step.get("action", "").lower()

            if isBrowsing and action_lower == "start_browsing_mode":
                logging.warning("LLM attempted to start browsing mode while it's already active.")
                print("Warning: LLM attempted to start browsing mode while it's already active.")
                continue

            if action_lower not in allowed_actions:
                logging.warning(f"Generated action '{step.get('action')}' is not allowed.")
                print(f"Warning: Generated action '{step.get('action')}' is not allowed.")
                continue

            # Additional validation for required fields
            if action_lower == "type" and not step.get("text"):
                logging.warning("Type action missing required 'text' field.")
                print("Warning: Type action missing required 'text' field.")
                continue

            if action_lower == "drag" and not step.get("target_description"):
                logging.warning("Drag action missing required 'target_description' field.")
                print("Warning: Drag action missing required 'target_description' field.")
                continue

            return [step]

        except Exception as e:
            logging.error(f"Error communicating with Qwen2.5: {e}")
            print(f"Error: Error communicating with Qwen2.5: {e}")
            continue

    logging.error("Failed to retrieve a valid step from the LLM after multiple attempts.")
    print("Error: Failed to retrieve a valid step from the LLM after multiple attempts.")
    return []

def generate_alternative_description(element_description, context_info):
    """
    Generate an alternative element description using the LLM.
    """
    prompt = (
        f"The automation script failed to locate the UI element with the description '{element_description}'.\n"
        f"Context: {context_info}\n"
        "Provide an alternative description for the UI element to improve detection accuracy.\n"
        "IMPORTANT: Output ONLY the alternative description as a plain string. No extra formatting."
    )

    payload = {
        "model": "huggingface.co/l33tkr3w/full:latest",
        "prompt": prompt,
        "temperature": 0.4
    }

    try:
        response = requests.post(QWEN_API_URL, json=payload, headers=HEADERS, stream=True)
        response.raise_for_status()
        json_fragments = []
        for line in response.iter_lines():
            if line:
                fragment_line = line.decode("utf-8").strip()
                try:
                    fragment_json = json.loads(fragment_line)
                    if "response" in fragment_json:
                        json_fragments.append(fragment_json["response"])
                except json.JSONDecodeError:
                    logging.warning(f"Skipped non-JSON fragment while generating description: {fragment_line}")
                    print(f"Skipped non-JSON fragment while generating description: {fragment_line}")
    
        full_response = "".join(json_fragments).strip()
        if full_response:
            logging.info(f"Generated alternative description: {full_response}")
            print(f"Generated alternative description: {full_response}")
            return full_response
        else:
            logging.error("LLM failed to generate an alternative description.")
            print("Error: LLM failed to generate an alternative description.")
            return None

    except Exception as e:
        logging.error(f"Error communicating with Qwen2.5 for alternative description: {e}")
        print(f"Error: Error communicating with Qwen2.5 for alternative description: {e}")
        return None

def draw_cursor_path(screenshot, start_pos, end_pos):
    """
    Draw a gradient line on the screenshot showing cursor movement from blue to red.
    """
    img = screenshot.copy()
    draw = ImageDraw.Draw(img)
    
    # Calculate line points for gradient
    num_points = 50  # Number of segments for smooth gradient
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = start_pos[0] + t * (end_pos[0] - start_pos[0])
        y = start_pos[1] + t * (end_pos[1] - start_pos[1])
        points.append((x, y))
    
    # Draw gradient line segments
    for i in range(len(points) - 1):
        t = i / (len(points) - 1)
        # Gradient from blue to red
        r = int(255 * t)
        b = int(255 * (1 - t))
        color = (r, 0, b)
        draw.line([points[i], points[i + 1]], fill=color, width=2)
    
    # Draw start and end markers
    marker_size = 5
    # Blue circle at start
    draw.ellipse([start_pos[0] - marker_size, start_pos[1] - marker_size,
                 start_pos[0] + marker_size, start_pos[1] + marker_size], 
                 fill=(0, 0, 255))
    # Red circle at end
    draw.ellipse([end_pos[0] - marker_size, end_pos[1] - marker_size,
                 end_pos[0] + marker_size, end_pos[1] + marker_size], 
                 fill=(255, 0, 0))
    
    return img

def check_action_success(pre_screenshot_path, post_screenshot_path, action, element_description, context):
    """
    Enhanced version that includes cursor movement visualization.
    """
    # Load the original post-action screenshot
    post_screenshot = Image.open(post_screenshot_path)
    
    # Get cursor positions from context
    start_pos = context.get('cursor_start_pos', (0, 0))
    end_pos = context.get('cursor_end_pos', (0, 0))
    
    # Draw cursor path on post-action screenshot
    post_with_cursor = draw_cursor_path(post_screenshot, start_pos, end_pos)
    
    # Save the modified screenshot
    cursor_path_screenshot = "screenshot_with_cursor_path.jpg"
    post_with_cursor.save(cursor_path_screenshot)
    
    prompt = f"""You are a computer vision system analyzing UI automation results.
Current Objective: {context.get('objective', '')}
Action Performed: {action} on '{element_description}'

The screenshots show before and after states. The second image includes a cursor movement visualization:
- Blue dot: Starting cursor position
- Red dot: Ending cursor position
- Blue-to-red gradient line: Cursor movement path

Analyze:
1. Cursor Movement:
   - Did the cursor reach its intended target?
   - Is the final position correct for the intended action?

2. UI Changes:
   - Are there visible changes around the cursor's end position?
   - Did the expected interaction occur at the target location?
   - Are there any error messages or unexpected changes?
   - If its a textbox or input field, state what was typed.

Output JSON only:
{{
    "status": "success" or "failure",
    "cursor_accuracy": 0-100,
    "visible_changes": [list specific UI changes],
    "cursor_position_correct": true/false,
    "error_messages": [any visible errors],
    "advice": "specific suggestion if failed."
}}"""

    pre_base64 = image_to_base64(pre_screenshot_path)
    post_base64 = image_to_base64(cursor_path_screenshot)

    if not pre_base64 or not post_base64:
        return {"status": "error", "advice": "Failed to encode one or both images."}

    payload = {
        "model": "minicpm-v",
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.4,
        "images": [pre_base64, post_base64]
    }

    try:
        response = requests.post(MINICPM_API_URL, json=payload, headers=HEADERS, stream=True)
        response.raise_for_status()
        json_fragments = []
        for line in response.iter_lines():
            if line:
                try:
                    fragment = json.loads(line.decode("utf-8"))
                    if "response" in fragment:
                        json_fragments.append(fragment["response"])
                except json.JSONDecodeError:
                    logging.warning(f"Skipped non-JSON fragment: {line.decode('utf-8')}")
                    print(f"Skipped non-JSON fragment: {line.decode('utf-8')}")

        full_response = "".join(json_fragments).strip().lower()

        match = re.search(r'\b(success|failure)\b', full_response)
        if match:
            status = match.group(1)
            if status == "success":
                return {"status": "success", "advice": ""}
            elif status == "failure":
                advice_match = re.search(r'failure[\s\S]*?:\s*(.+)', full_response)
                advice = advice_match.group(1).strip() if advice_match else "No advice provided."
                return {"status": "failure", "advice": advice}
        else:
            return {"status": "failure", "advice": "Unclear response from vision model."}

    except Exception as e:
        logging.error(f"Error communicating with MiniCPM-V for action success: {e}")
        print(f"Error: Error communicating with MiniCPM-V for action success: {e}")
        return {"status": "error", "advice": "Error in MiniCPM-V processing."}

def get_vision_advice(screenshot_path, step, objective):
    """
    Fetch concise advice from the Vision model to help adjust the strategy.
    """
    prompt = (
        "You are part of an automated task system working alongside a language model. "
        "The system's objective is:\n"
        f"{objective}\n\n"
        "Analyze the screenshot and decide if the last action was successful. If not, describe the current screen state.\n"
        "Output JSON:\n"
        "{\n"
        "  \"status\": \"success\" or \"failure\",\n"
        "  \"advice\": \"concise info on the action status.\"\n"
        "}\n"
        "DO NOT use markdown or code fences. Just output JSON."
    )

    base64_image = image_to_base64(screenshot_path)
    if not base64_image:
        return "Unable to encode image for vision advice."

    payload = {
        "model": "minicpm-v",
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.4,
        "images": [base64_image]
    }

    try:
        response = requests.post(MINICPM_API_URL, json=payload, headers=HEADERS, stream=True)
        response.raise_for_status()
        json_fragments = []
        for line in response.iter_lines():
            if line:
                try:
                    fragment = json.loads(line.decode("utf-8"))
                    if "response" in fragment:
                        json_fragments.append(fragment["response"])
                except json.JSONDecodeError:
                    logging.warning(f"Skipped non-JSON fragment while generating description: {line.decode('utf-8')}")
                    print(f"Skipped non-JSON fragment while generating description: {line.decode('utf-8')}")

        full_response = "".join(json_fragments).strip()

        try:
            feedback = json.loads(full_response)
            return feedback.get("advice", "No advice provided.") if feedback.get("status", "failure") == "failure" else "success"
        except json.JSONDecodeError:
            logging.error(f"Failed to parse vision model response: {full_response}")
            return "Unclear response from vision model."

    except Exception as e:
        logging.error(f"Error communicating with MiniCPM-V for advice: {e}")
        print(f"Error: Error communicating with MiniCPM-V for advice: {e}")
        return "Unable to obtain advice from the Vision model."

def locate_element_with_template(template_path, confidence=0.8):
    """
    Locate an element on the screen using an image template.
    """
    if not os.path.exists(template_path):
        logging.error(f"Template image '{template_path}' does not exist.")
        print(f"Error: Template image '{template_path}' does not exist.")
        return None

    try:
        location = pyautogui.locateCenterOnScreen(template_path, confidence=confidence)
        if location:
            logging.info(f"Element located at ({location.x}, {location.y}) using template.")
            print(f"Element located at ({location.x}, {location.y}) using template.")
            return {"x": location.x, "y": location.y}
        else:
            logging.warning(f"Element not found using template {template_path}.")
            print(f"Element not found using template {template_path}.")
            return None
    except Exception as e:
        logging.error(f"Error during template matching: {e}")
        print(f"Error during template matching: {e}")
        return None

def detect_coordinates(element_description, screenshot_path):
    screenshot = Image.open(screenshot_path).convert("RGB")
    screen_width, screen_height = pyautogui.size()
    img_width, img_height = screenshot.size

    prompt = f"<OPEN_VOCABULARY_DETECTION> {element_description}"
    inputs = pta_processor(
        text=prompt,
        images=screenshot,
        return_tensors="pt"
    ).to(device, torch_dtype)

    generated_ids = pta_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )

    result = pta_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_result = pta_processor.post_process_generation(
        result,
        task="<OPEN_VOCABULARY_DETECTION>",
        image_size=(screenshot.width, screenshot.height)
    )

    detection_data = parsed_result.get("<OPEN_VOCABULARY_DETECTION>", {})
    polygons = detection_data.get("polygons", [])
    polygon_labels = detection_data.get("polygons_labels", [])

    scale_x = screen_width / img_width
    scale_y = screen_height / img_height

    if polygons:
        polygon = polygons[0]
        points = []
        for i in range(0, len(polygon[0]), 2):
            x = polygon[0][i] * scale_x
            y = polygon[0][i + 1] * scale_y
            points.append((x, y))

        unique_points = list(set(points))

        if len(unique_points) < 2:
            return None

        x_coords = [point[0] for point in unique_points]
        y_coords = [point[1] for point in unique_points]
        x_center = sum(x_coords) / len(x_coords)
        y_center = sum(y_coords) / len(y_coords)
        
        return {"x": x_center, "y": y_center, "polygon_labels": polygon_labels, "polygon": unique_points}
    else:
        return None

def smooth_move_to(x, y, context):
    """Smoothly move the cursor and track its position."""
    # Get current position
    current_x, current_y = pyautogui.position()
    context['cursor_start_pos'] = (current_x, current_y)
    
    # Move cursor
    duration = 0.5
    pyautogui.moveTo(x, y, duration=duration)
    
    # Store end position
    context['cursor_end_pos'] = (x, y)

def execute_action(context, step, screenshot_path, retry_count=3):
    """
    Execute a UI action based on the provided step.
    Updated to include breadcrumbs tracking.
    """
    action = step["action"].lower()

    # Store initial cursor position
    start_x, start_y = pyautogui.position()
    context['cursor_start_pos'] = (start_x, start_y)

    if action == "no_action":
        logging.info("No further action needed. Objective may be complete.")
        print("No further action needed. Objective may be complete.")
        return True

    if action == "start_browsing_mode":
        url = step.get("text", "").strip()
        if not url:
            logging.error("No URL provided for start_browsing_mode action.")
            print("Error: No URL provided for start_browsing_mode action.")
            return False

        success = start_browsing_mode(url)
        if success:
            logging.info(f"Browsing mode started with URL: {url}")
            print(f"Browsing mode started with URL: {url}")
            context['completed_steps'].append(step)
            update_breadcrumbs(context, step)  # Update breadcrumbs
            return True
        else:
            logging.error(f"Failed to start browsing mode with URL '{url}'.")
            print(f"Failed to start browsing mode with URL '{url}'.")
            return False

    elif action == "stop_browsing_mode":
        success = stop_browsing_mode()
        if success:
            logging.info("Browsing mode stopped successfully.")
            print("Browsing mode stopped successfully.")
            context['completed_steps'].append(step)
            update_breadcrumbs(context, step)  # Update breadcrumbs
            return True
        else:
            logging.error("Failed to stop browsing mode.")
            print("Failed to stop browsing mode.")
            return False

    # Attempt to locate the UI element with retries
    for attempt in range(1, retry_count + 1):
        coords = detect_coordinates(step["element_description"], screenshot_path)
        if coords:
            x, y = coords["x"], coords["y"]
            break
        else:
            logging.warning(f"Attempt {attempt}/{retry_count}: Failed to locate element '{step['element_description']}'.")
            print(f"Attempt {attempt}/{retry_count}: Failed to locate element '{step['element_description']}'.")
            if attempt == retry_count:
                template_path = f"templates/{step['element_description'].replace(' ', '_')}_template.png"
                coords = locate_element_with_template(template_path)
                if coords:
                    logging.info(f"Located '{step['element_description']}' using template matching.")
                    print(f"Located '{step['element_description']}' using template matching.")
                    break
                else:
                    logging.error(f"Failed to locate element '{step['element_description']}' using both PTA-1 and template matching.")
                    print(f"Failed to locate element '{step['element_description']}' using both PTA-1 and template matching.")
                    context_info = "Unable to detect the element with the current description."
                    alternative_description = generate_alternative_description(step["element_description"], context_info)
                    if alternative_description:
                        step["element_description"] = alternative_description
                        logging.info(f"Retrying with alternative description: {alternative_description}")
                        print(f"Retrying with alternative description: {alternative_description}")
                        coords = detect_coordinates(step["element_description"], screenshot_path)
                        if coords:
                            x, y = coords["x"], coords["y"]
                            break
                        else:
                            template_path = f"templates/{step['element_description'].replace(' ', '_')}_template.png"
                            coords = locate_element_with_template(template_path)
                            if coords:
                                logging.info(f"Located '{step['element_description']}' using template matching with alternative description.")
                                print(f"Located '{step['element_description']}' using template matching with alternative description.")
                                break
                    logging.error(f"Failed to locate element '{step['element_description']}' after generating alternative description.")
                    print(f"Failed to locate element '{step['element_description']}' after generating alternative description.")
                    return False
        sleep(2)

    # Execute the action with cursor tracking
    try:
        if action == "click":
            smooth_move_to(x, y, context)
            pyautogui.click()
            context['cursor_end_pos'] = (x, y)
            logging.info(f"Clicked on '{step['element_description']}' at ({x}, {y}).")
            print(f"Clicked on '{step['element_description']}' at ({x}, {y}).")

        elif action == "doubleclick":
            smooth_move_to(x, y, context)
            pyautogui.doubleClick()
            context['cursor_end_pos'] = (x, y)
            logging.info(f"Double-clicked on '{step['element_description']}' at ({x}, {y}).")
            print(f"Double-clicked on '{step['element_description']}' at ({x}, {y}).")

        elif action == "rightclick":
            smooth_move_to(x, y, context)
            pyautogui.rightClick()
            context['cursor_end_pos'] = (x, y)
            logging.info(f"Right-clicked on '{step['element_description']}' at ({x}, {y}).")
            print(f"Right-clicked on '{step['element_description']}' at ({x}, {y}).")

        elif action == "drag":
            target_description = step.get("target_description", "").strip()
            if not target_description:
                logging.error("The 'drag' action requires a 'target_description'.")
                print("Error: The 'drag' action requires a 'target_description'.")
                return False

            target_coords = detect_coordinates(target_description, screenshot_path)
            if not target_coords:
                template_path = f"templates/{target_description.replace(' ', '_')}_template.png"
                target_coords = locate_element_with_template(template_path)
                if not target_coords:
                    logging.error(f"Failed to locate target element '{target_description}' for drag action.")
                    print(f"Failed to locate target element '{target_description}' for drag action.")
                    return False

            target_x, target_y = target_coords["x"], target_coords["y"]
            
            smooth_move_to(x, y, context)
            pyautogui.dragTo(target_x, target_y, duration=0.5, button='left')
            context['cursor_end_pos'] = (target_x, target_y)
            logging.info(f"Dragged from '{step['element_description']}' at ({x}, {y}) to '{target_description}' at ({target_x}, {target_y}).")
            print(f"Dragged from '{step['element_description']}' at ({x}, {y}) to '{target_description}' at ({target_x}, {target_y}).")

        elif action == "hover":
            smooth_move_to(x, y, context)
            context['cursor_end_pos'] = (x, y)
            logging.info(f"Hovered over '{step['element_description']}' at ({x}, {y}).")
            print(f"Hovered over '{step['element_description']}' at ({x}, {y}).")

        elif action == "move":
            smooth_move_to(x, y, context)
            context['cursor_end_pos'] = (x, y)
            logging.info(f"Moved cursor to '{step['element_description']}' at ({x}, {y}) without clicking.")
            print(f"Moved cursor to '{step['element_description']}' at ({x}, {y}) without clicking.")

        elif action == "type":
            text = step.get("text", "")
            if not text:
                logging.error(f"No text provided for '{action}' action.")
                print(f"Error: No text provided for '{action}' action.")
                return False

            smooth_move_to(x, y, context)
            pyautogui.click()
            context['cursor_end_pos'] = (x, y)
            pyautogui.write(text, interval=0.05)
            logging.info(f"Typed '{text}' into '{step['element_description']}' at ({x}, {y}).")
            print(f"Typed '{text}' into '{step['element_description']}' at ({x}, {y}).")
            
            # If this is a search field, press Enter
            if "search" in step["element_description"].lower():
                sleep(0.5)  # Small delay before Enter
                pyautogui.press('enter')
                logging.info("Pressed Enter after search input")
                print("Pressed Enter after search input")
                sleep(2)  # Wait for search results

        else:
            logging.error(f"Unknown action '{action}'.")
            print(f"Error: Unknown action '{action}'.")
            return False

    except Exception as e:
        logging.error(f"Exception occurred while executing action '{action}': {e}")
        print(f"Exception occurred while executing action '{action}': {e}")
        return False

    # After the action, capture the post-action screenshot with cursor path
    if action in ["click", "doubleclick", "rightclick", "drag", "hover", "move", "type"] and screenshot_path:
        post_action_screenshot = "screenshot_after_action.jpg"
        success_capture_after = capture_entire_screen_with_blackout(post_action_screenshot)
        if success_capture_after:
            vision_result = check_action_success(screenshot_path, post_action_screenshot, step["action"], 
                                               step["element_description"], context)
            print(f"Vision Feedback: {vision_result}")
            logging.debug(f"Vision Feedback: {vision_result}")

            if vision_result["status"] == "success":
                context['completed_steps'].append(step)
                update_breadcrumbs(context, step)  # Update breadcrumbs
                logging.info(f"Step marked as completed based on vision feedback: {step['action']} on '{step['element_description']}'")
                print(f"Step marked as completed based on vision feedback: {step['action']} on '{step['element_description']}'")
            elif vision_result["status"] == "failure":
                logging.warning("Feedback indicates a failure. Consulting the vision model for advice...")
                print("Feedback indicates a failure. Consulting the vision model for advice...")

                advice = vision_result.get("advice", "No advice provided.")
                print(f"Vision Model Advice: {advice}")
                logging.debug(f"Vision Model Advice: {advice}")

                if advice != "No advice provided.":
                    context['objective'] = f"{context['objective']}\nNote: {advice}"
            else:
                logging.error("Encountered an error with vision feedback.")
                print("Encountered an error with vision feedback.")
        else:
            # If post-action screenshot fails, we can't do a vision check, but we can still consider it completed.
            context['completed_steps'].append(step)
            update_breadcrumbs(context, step)  # Update breadcrumbs
    else:
        # If no vision scenario applies or no screenshot_path, consider step completed
        context['completed_steps'].append(step)
        update_breadcrumbs(context, step)  # Update breadcrumbs

    return True


def get_vision_advice_for_stuck_state(screenshot_path, context, repeated_steps):
    """
    Get advice from vision model when we're stuck in a loop.
    Vision model ONLY handles image content analysis, not UI elements.
    """
    last_action = context['completed_steps'][-1]['action'].lower() if context.get('completed_steps') else None

    # If we're looking for specific image content
    prompt = f"""You are helping with web automation that seems stuck.
Current objective: {context.get('objective', '')}

Analyze the image content in the screenshot (ignore UI elements like menus and buttons):
1. Are the requested images visible? (e.g., yellow duck, red car, etc.)
2. What are their visual characteristics?
3. Where are they located relative to each other?

Output JSON only:
{{
    "target_images": [
        {{
            "description": "visual description of the image",
            "matches_criteria": true/false,
            "location": "general location in the view (e.g., 'top left', 'center')",
            "distinctive_features": ["list of notable visual features"]
        }}
    ],
    "confidence": 0-100
}}"""

    base64_image = image_to_base64(screenshot_path)
    if not base64_image:
        return None

    payload = {
        "model": "minicpm-v",
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.4,
        "images": [base64_image]
    }

    try:
        response = requests.post(MINICPM_API_URL, json=payload, headers=HEADERS, stream=True)
        response.raise_for_status()
        
        json_fragments = []
        for line in response.iter_lines():
            if line:
                try:
                    fragment = json.loads(line.decode("utf-8"))
                    if "response" in fragment:
                        json_fragments.append(fragment["response"])
                except json.JSONDecodeError:
                    continue

        full_response = "".join(json_fragments).strip()
        
        try:
            feedback = json.loads(full_response)
            logging.info(f"Vision advice for stuck state: {feedback}")
            print(f"\nVision Analysis of Stuck State:")
            print(f"Current State: {feedback.get('current_state', '')}")
            print(f"Search Results Visible: {feedback.get('search_results_visible', False)}")
            print(f"Recommended Action: {feedback.get('recommended_action', {}).get('description', '')}")
            print(f"Advice: {feedback.get('advice', '')}\n")
            
            # Update context with vision feedback
            context['current_screen_state'] = feedback.get('current_state')
            context['clickable_elements'] = feedback.get('clickable_elements', [])
            if feedback.get('recommended_action'):
                context['next_action_suggestion'] = feedback['recommended_action']
            
            return feedback
            
        except json.JSONDecodeError:
            logging.error("Failed to parse vision model response")
            return None

    except Exception as e:
        logging.error(f"Error getting vision advice: {e}")
        return None

def get_vision_guidance(screenshot_path, query, context):
    """
    Ask vision model to locate visual elements and return coordinates.
    """
    prompt = f"""Analyze this screenshot and locate: {query}
Focus ONLY on providing coordinates for the target.

Output JSON with ONLY:
{{
    "found": true/false,
    "coordinates": {{
        "x": pixel position from left,
        "y": pixel position from top
    }},
    "confidence": 0-100
}}"""

    base64_image = image_to_base64(screenshot_path)
    if not base64_image:
        return None

    payload = {
        "model": "minicpm-v",
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.4,
        "images": [base64_image]
    }

    try:
        response = requests.post(MINICPM_API_URL, json=payload, headers=HEADERS, stream=True)
        response.raise_for_status()
        
        json_fragments = []
        for line in response.iter_lines():
            if line:
                try:
                    fragment = json.loads(line.decode("utf-8"))
                    if "response" in fragment:
                        json_fragments.append(fragment["response"])
                except json.JSONDecodeError:
                    continue

        full_response = "".join(json_fragments).strip()
        feedback = json.loads(full_response)
        
        # Just return the raw coordinates if found
        if feedback.get("found") and feedback.get("coordinates"):
            return feedback["coordinates"]
        
        return None

    except Exception as e:
        logging.error(f"Error getting vision coordinates: {e}")
        return None


def detect_coordinates_with_vision_fallback(element_description, screenshot_path, context):
    """
    Enhanced detection that uses:
    - PTA-1 for UI elements (menus, buttons, text)
    - Vision model for image content (when looking for specific images)
    """
    # Check if we're looking for a UI element
    is_ui_element = any(term in element_description.lower() 
                       for term in ["menu", "button", "click", "input", "text", "open", "window"])
    
    if is_ui_element:
        # Use PTA-1 for UI elements
        coords = detect_coordinates(element_description, screenshot_path)
        if coords:
            return coords, "pta"
        return None, None
        
    # If we're looking for specific image content
    visual_terms = ["yellow", "red", "duck", "first", "second", "animal", "photo"]
    if any(term in element_description.lower() for term in visual_terms):
        vision_guidance = get_vision_guidance(screenshot_path, element_description, context)
        if vision_guidance and vision_guidance.get("found"):
            best_match = max(vision_guidance["locations"], 
                           key=lambda x: x["confidence"])
            
            context["vision_guidance"] = vision_guidance
            return {
                "x": best_match.get("x", 0),
                "y": best_match.get("y", 0),
                "needs_visual_refinement": True
            }, "vision"
            
    return None, None


def convert_relative_position_to_coordinates(position_description, screenshot_path):
    """
    Convert a relative position description from vision model into actual x,y coordinates.
    Uses PTA-1 to detect reference UI elements and calculates position from there.
    """
    screenshot = Image.open(screenshot_path)
    screen_width, screen_height = screenshot.size
    
    # Common position keywords and their approximate screen positions
    position_mappings = {
        "top": (0.5, 0.2),
        "bottom": (0.5, 0.8),
        "left": (0.2, 0.5),
        "right": (0.8, 0.5),
        "center": (0.5, 0.5),
        "top left": (0.2, 0.2),
        "top right": (0.8, 0.2),
        "bottom left": (0.2, 0.8),
        "bottom right": (0.8, 0.8)
    }
    
    # Extract numbers for nth element (first, second, etc.)
    ordinal_mapping = {
        "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
        "last": -1
    }
    
    try:
        # First, try to detect any UI elements mentioned as reference points
        reference_elements = []
        reference_coords = None
        
        # Look for UI elements in the description
        desc_lower = position_description.lower()
        if "next to" in desc_lower or "near" in desc_lower or "below" in desc_lower or "above" in desc_lower:
            # Extract potential UI element descriptions
            for part in desc_lower.split():
                if part not in ["next", "to", "near", "below", "above", "the", "of"]:
                    # Try to detect this as a UI element
                    coords = detect_coordinates(part, screenshot_path)
                    if coords:
                        reference_elements.append((part, coords))
        
        if reference_elements:
            # Use the first detected reference element
            reference_name, reference_coords = reference_elements[0]
            base_x, base_y = reference_coords["x"], reference_coords["y"]
            
            # Adjust based on relative position words
            offset = 50  # pixels
            if "below" in desc_lower:
                return {"x": base_x, "y": base_y + offset}
            elif "above" in desc_lower:
                return {"x": base_x, "y": base_y - offset}
            elif "right" in desc_lower:
                return {"x": base_x + offset, "y": base_y}
            elif "left" in desc_lower:
                return {"x": base_x - offset, "y": base_y}
            
        # If no UI reference points, use general position mapping
        for pos, (x_ratio, y_ratio) in position_mappings.items():
            if pos in desc_lower:
                return {
                    "x": int(screen_width * x_ratio),
                    "y": int(screen_height * y_ratio)
                }
        
        # Check for ordinal numbers (first, second, etc.)
        for ordinal, index in ordinal_mapping.items():
            if ordinal in desc_lower:
                # For items in a list or grid, estimate position based on index
                base_x = screen_width * 0.2  # Start from 20% in
                base_y = screen_height * 0.3  # Start from 30% down
                item_width = screen_width * 0.15  # Each item takes 15% of width
                
                if index == -1:  # "last" - assume it's far right
                    x = screen_width * 0.8
                else:
                    x = base_x + (item_width * index)
                    
                return {"x": int(x), "y": int(base_y)}
        
        # If no specific position info found, aim for the center of the screen
        return {
            "x": int(screen_width * 0.5),
            "y": int(screen_height * 0.5)
        }
            
    except Exception as e:
        logging.error(f"Error converting position to coordinates: {e}")
        # Default to screen center if conversion fails
        return {
            "x": int(screen_width * 0.5),
            "y": int(screen_height * 0.5)
        }

def execute_action_with_vision_guidance(context, step, coords, screenshot_path):
    """
    Execute action with additional visual guidance for non-UI elements.
    """
    guidance = context.get("vision_guidance", {})
    if not guidance:
        return execute_action(context, step, screenshot_path)
        
    best_match = max(guidance.get("locations", []), 
                    key=lambda x: x.get("confidence", 0))
    
    # Convert the vision guidance into actual coordinates
    refined_coords = convert_relative_position_to_coordinates(
        best_match["suggested_click_point"],
        screenshot_path
    )
    
    # Update the coordinates in the original coords dict
    coords.update(refined_coords)
    
    # Remove the visual refinement flag
    if "needs_visual_refinement" in coords:
        del coords["needs_visual_refinement"]
    
    # Log the refined coordinates
    logging.info(f"Refined coordinates based on vision guidance: ({coords['x']}, {coords['y']})")
    print(f"Refined coordinates based on vision guidance: ({coords['x']}, {coords['y']})")
    
    # Execute the action with the refined coordinates
    step["element_description"] = best_match.get("description", step["element_description"])
    return execute_action(context, step, screenshot_path)


def capture_entire_screen_with_blackout(screenshot_path):
    """
    Capture the entire screen and blackout any Anaconda windows present.
    """
    try:
        screenshot = pyautogui.screenshot().convert("RGB")
        logging.info("Captured the entire screen for screenshot.")
        print("Captured the entire screen for screenshot.")

        desktop = Desktop(backend="win32")

        for window in desktop.windows():
            if "Anaconda" in window.window_text():
                rect = window.rectangle()
                logging.info(f"Found Anaconda window: {window.window_text()} at {rect}")
                print(f"Found Anaconda window: {window.window_text()} at {rect}")

                draw = ImageDraw.Draw(screenshot)
                draw.rectangle([rect.left, rect.top, rect.right, rect.bottom], fill="black")
                logging.info(f"Blacked out Anaconda window at ({rect.left}, {rect.top}, {rect.right}, {rect.bottom}).")
                print(f"Blacked out Anaconda window at ({rect.left}, {rect.top}, {rect.right}, {rect.bottom}).")

        screenshot.save(screenshot_path)
        logging.info(f"Saved entire screen screenshot with Anaconda window blacked out to {screenshot_path}.")
        print(f"Saved entire screen screenshot with Anaconda window blacked out to {screenshot_path}.")
        return True
    except Exception as e:
        logging.error(f"Failed to capture entire screen with blackout: {e}")
        print(f"Error: Failed to capture entire screen with blackout: {e}")
        return False

def main():
    # Initialize context
    context = {
        'objective': '',
        'completed_steps': [],
        'skipped_steps': [],
        'use_vision': True,
        'current_screen_state': None,
        'clickable_elements': [],
        'next_action_suggestion': None
    }

    failed_attempts = 0
    max_failed_attempts = 25
    repeated_steps = 0
    max_repeated_steps = 25

    post_action_delay = 1

    vision_enabled_input = input("Enable vision model? (yes/no): ").strip().lower()
    context['use_vision'] = vision_enabled_input == "yes"
    context['objective'] = input("Enter the objective: ")

    while failed_attempts < max_failed_attempts and repeated_steps < max_repeated_steps:
        try:
            print("\nFetching the next step...")
            logging.info("\nFetching the next step...")

            steps = get_next_step(context)

            if not steps:
                logging.info("No steps generated. Exiting.")
                print("No steps generated. Exiting.")
                break

            step = steps[0]

            # Check if step was already completed
            if any(
                (completed_step["action"].lower() == step["action"].lower() and
                 completed_step.get("text", "").lower() == step.get("text", "").lower()) if "text" in step else
                (completed_step["action"].lower() == step["action"].lower() and
                 completed_step.get("element_description", "").lower() == step.get("element_description", "").lower())
                for completed_step in context['completed_steps']
            ):
                context['skipped_steps'].append(step)
                repeated_steps += 1
                
                # After a few repeats, get vision guidance
                if repeated_steps >= 3:
                    current_screenshot = "current_state_screenshot.jpg"
                    if capture_entire_screen_with_blackout(current_screenshot):
                        vision_feedback = get_vision_advice_for_stuck_state(
                            current_screenshot, 
                            context,
                            repeated_steps
                        )
                        if vision_feedback:
                            # Update the objective with the vision model's advice
                            if vision_feedback.get('advice'):
                                context['objective'] = f"{context['objective']}\nVision Guidance: {vision_feedback['advice']}"
                            
                            # Reset repeated steps counter since we got new guidance
                            repeated_steps = 0

                logging.info(f"Step already completed. Skipping. (Repeated {repeated_steps}/{max_repeated_steps})")
                print(f"Step already completed. Skipping. (Repeated {repeated_steps}/{max_repeated_steps})")
                continue
            else:
                repeated_steps = 0

            print(f"Processing step: {step}")
            logging.info(f"Processing step: {step}")

            if step["action"].lower() not in ["start_browsing_mode", "stop_browsing_mode", "no_action"]:
                pre_action_screenshot = "screenshot_before_action.jpg"
                success_capture_before = capture_entire_screen_with_blackout(pre_action_screenshot)
                if not success_capture_before:
                    failed_attempts += 1
                    logging.error("Pre-action screenshot capture failed.")
                    print("Error: Pre-action screenshot capture failed.")
                    continue

                # Try to detect coordinates with vision fallback
                coords, detection_method = detect_coordinates_with_vision_fallback(
                    step["element_description"], 
                    pre_action_screenshot, 
                    context
                )
                
                if coords:
                    if detection_method == "vision":
                        success = execute_action_with_vision_guidance(
                            context, step, coords, pre_action_screenshot
                        )
                    else:
                        success = execute_action(context, step, pre_action_screenshot)
                else:
                    success = execute_action(context, step, pre_action_screenshot)

            else:
                success = execute_action(context, step, None)

            if success:
                logging.info(f"Step '{step['action']}' on '{step.get('element_description', '')}' executed.")
                print(f"Step '{step['action']}' on '{step.get('element_description', '')}' executed.")
            else:
                failed_attempts += 1
                logging.error(f"Action execution failed: {step['action']} on '{step.get('element_description', '')}' (Attempt {failed_attempts}/{max_failed_attempts})")
                print(f"Action execution failed: {step['action']} on '{step.get('element_description', '')}' (Attempt {failed_attempts}/{max_failed_attempts})")
                continue

            if context['use_vision'] and step["action"].lower() not in ["start_browsing_mode", "stop_browsing_mode", "no_action"]:
                sleep(post_action_delay)
                logging.debug(f"Waiting {post_action_delay} seconds for UI to update.")
                print(f"Waiting {post_action_delay} seconds for UI to update.")

            sleep(1)

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            print(f"Error: An error occurred: {e}")
            failed_attempts += 1

    logging.info("\nAutomation process completed or reached maximum failed attempts.")
    print("\nAutomation process completed or reached maximum failed attempts.")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
