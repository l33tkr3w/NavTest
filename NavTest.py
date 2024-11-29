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
        sleep(4)  # Wait for 5 seconds to allow the browser to load completely
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

def prepare_prompt(context):
    """
    Prepare the prompt for the Language Model based on the current context.

    Args:
        context (dict): A dictionary containing 'objective', 'completed_steps', and 'skipped_steps'.

    Returns:
        str: The constructed prompt.
    """
    completed_steps = context.get('completed_steps', [])
    skipped_steps = context.get('skipped_steps', [])
    objective = context.get('objective', '')
    use_vision = context.get('use_vision', True)

    completed_steps_text = ""
    skipped_steps_text = ""
    if completed_steps:
        if use_vision:
            completed_steps_text = (
                "The following steps have already been completed:\n" +
                "\n".join([
                    f"- Action: {step['action']}, Element: {step['element_description']}, Target: {step.get('target_description', '')}, Text: {step.get('text', '')}"
                    for step in completed_steps
                ]) +
                "\n"
            )
    if skipped_steps:
        if use_vision:
            skipped_steps_text = (
                "The following steps were suggested but have been skipped because they were already completed:\n" +
                "\n".join([
                    f"- Action: {step['action']}, Element: {step['element_description']}, Target: {step.get('target_description', '')}, Text: {step.get('text', '')}"
                    for step in skipped_steps
                ]) +
                "\n"
            )

    browsing_state = "ACTIVE" if isBrowsing else "INACTIVE"

    browsing_instructions = (
        "- To browse the internet, use the 'start_browsing_mode' function with a URL. Example:\n"
        "  {\n"
        "    \"action\": \"start_browsing_mode\",\n"
        "    \"element_description\": \"\",\n"
        "    \"target_description\": \"\",\n"
        "    \"text\": \"https://www.example.com\"\n"
        "  }\n"
        "- The 'text' field must contain a valid URL.\n"
        "- Once browsing mode is started, do not attempt to start it again unless you have ended it. Continue using mouse and keyboard actions if it's active.\n"
        "- When finished browsing, use 'stop_browsing_mode' to end browsing mode.\n"
        "- While browsing mode is active, only the current browser window is processed.\n"
        "- If posting on Reddit, you must click on 'Create Post'. Then enter a title into the 'Title field' and your message into the 'Body field'.\n"
    )

    # Define the list of allowed actions
    allowed_actions = [
        "click",
        "doubleclick",
        "rightclick",
        "drag",
        "hover",
        "move",
        "type",
        "enter text into input field",
        "start_browsing_mode",
        "stop_browsing_mode"
    ]

    return (
        f"You are an intelligent assistant responsible for guiding a Windows 11 "
        f"application in completing tasks.\n"
        f"The application allows users to interact with UI elements (e.g., clicking, "
        f"typing, clicking and dragging) using a specialized UI element detection model to detect target coordinates based on description.\n"
        f"Your role is to provide **only the next atomic step** needed to achieve "
        f"the objective step-by-step.\n\n"
        f"{completed_steps_text}"
        f"{skipped_steps_text}"
        f"Objective: {objective}\n\n"
        f"Browsing State: {browsing_state}\n\n"
        f"{browsing_instructions}\n\n"
        "Instructions:\n"
        f"- Provide the **next atomic step** to complete the objective. Use only the following actions: {', '.join([f'\'{action.capitalize()}\'' for action in allowed_actions])}.\n"
        "- Each step must include:\n"
        "  - **action**: The specific task to perform (e.g., 'Click', 'Type', 'Enter text into input field', 'drag', 'move').\n"
        "  - **element_description**: Name the element to target. Example: 'Title field'.\n"
        "  - **target_description** (if applicable): Name the target element for drag actions. Example: 'Folder icon'.\n"
        "  - **text** (if applicable): For 'Type' actions or browsing actions, provide the text to be entered (e.g., the URL for browsing).\n\n"
        "Output format (JSON):\n"
        "{\n"
        "  \"action\": \"<action>\",\n"
        "  \"element_description\": \"<element_description>\",\n"
        "  \"target_description\": \"<target_description (if applicable)>\",\n"
        "  \"text\": \"<text (if applicable)>\"\n"
        "}"
    )

def get_next_step(context, retry_limit=5):
    """
    Fetch the next step from the Language Model based on the current context.

    Args:
        context (dict): A dictionary containing 'objective', 'completed_steps', and 'skipped_steps'.
        retry_limit (int): Number of attempts to fetch a valid step.

    Returns:
        list: A list containing the next step as a dictionary, or empty list if failed.
    """
    allowed_actions = [
        "click",
        "doubleclick",
        "rightclick",
        "drag",
        "hover",
        "move",
        "type",
        "enter text into input field",
        "start_browsing_mode",
        "stop_browsing_mode"
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
            json_fragments = []
            for line in response.iter_lines():
                if line:
                    try:
                        fragment = line.decode("utf-8").strip()
                        # Remove triple backticks and parse JSON
                        if fragment.startswith("```"):
                            fragment = fragment.replace("```json", "").replace("```", "").strip()
                        fragment_json = json.loads(fragment)
                        if "response" in fragment_json:
                            json_fragments.append(fragment_json["response"])
                    except json.JSONDecodeError:
                        logging.warning(f"Skipped non-JSON fragment: {line.decode('utf-8')}")
                        print(f"Skipped non-JSON fragment: {line.decode('utf-8')}")
        
            full_response = "".join(json_fragments).strip()
            if not full_response:
                logging.warning("Empty response from LLM.")
                print("Warning: Empty response from LLM.")
                continue  # Retry if response is empty

            # Remove triple backticks and clean the response
            if full_response.startswith("```"):
                full_response = full_response.replace("```json", "").replace("```", "").strip()

            # Parse JSON
            step = json.loads(full_response)
            action_lower = step.get("action", "").lower()

            # Additional validation based on browsing state
            if isBrowsing and action_lower == "start_browsing_mode":
                logging.warning("LLM attempted to start browsing mode while it's already active.")
                print("Warning: LLM attempted to start browsing mode while it's already active.")
                continue  # Retry fetching a valid step

            if action_lower not in allowed_actions:
                logging.warning(f"Generated action '{step.get('action')}' is not allowed.")
                print(f"Warning: Generated action '{step.get('action')}' is not allowed.")
                continue  # Retry if action is not allowed

            # If valid, return the step
            return [step]

        except Exception as e:
            logging.error(f"Error communicating with Qwen2.5: {e}")
            print(f"Error: Error communicating with Qwen2.5: {e}")
            continue  # Retry on exception

    # If all retries failed
    logging.error("Failed to retrieve a valid step from the LLM after multiple attempts.")
    print("Error: Failed to retrieve a valid step from the LLM after multiple attempts.")
    return []



def check_action_success(pre_screenshot_path, post_screenshot_path, action, element_description):
    """
    Determine if an action was successful by comparing pre-action and post-action screenshots.
    
    Args:
        pre_screenshot_path (str): Path to the pre-action screenshot.
        post_screenshot_path (str): Path to the post-action screenshot.
        action (str): The action that was performed.
        element_description (str): Description of the UI element involved in the action.
    
    Returns:
        dict: Contains 'status' ('success' or 'failure') and 'advice' if failed.
    """
    prompt = (
        "You are part of an application used for automation, a LLM is performing actions and your feedback assists in step completion.\n"
        f"Action Performed: {action} on '{element_description}'.\n"
        "Based on the Left pre-action and Right post-action screenshots, does it seem like the action completed successfully? If they are the same, suggest continuing to next step. "
        "Reply with 'success' or 'failure' only.\n"
        "If 'failure', provide a brief suggestion of a better action to assist the model."
    )
    
    # Convert images to base64
    pre_base64 = image_to_base64(pre_screenshot_path)
    post_base64 = image_to_base64(post_screenshot_path)
    
    if not pre_base64 or not post_base64:
        return {"status": "error", "advice": "Failed to encode one or both images."}
    
    payload = {
        "model": "minicpm-v",
        "prompt": prompt,
        "max_tokens": 2048,  # Adjust as needed
        "temperature": 0.4,  # Lower temperature for deterministic output
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
    
        # Extract 'success' or 'failure' using regex
        match = re.search(r'\b(success|failure)\b', full_response)
        if match:
            status = match.group(1)
            if status == "success":
                return {"status": "success", "advice": ""}
            elif status == "failure":
                # Extract the advice sentence
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

    Args:
        screenshot_path (str): Path to the current screenshot.
        step (dict): The last step attempted by the LLM.
        objective (str): The overarching objective of the task.

    Returns:
        str: Concise advice or feedback from the vision model.
    """
    prompt = (
        "You are part of an automated task system working alongside a language model. "
        "The system's objective is as follows:\n"
        f"Objective: {objective}\n\n"
        "Your role is to analyze screenshots and provide feedback to determine whether the last action was successful. "
        "You will use the context of the attempted action and the current screen state to give precise feedback.\n\n"
        "If the task is posting on reddit, do not suggest clicking the Create Post button. The model instead needs to enter the title and body data.\n\n"
        "Here are the details of the last attempted step:\n"
        f"- Action: {step.get('action', 'N/A')}\n"
        f"- Element Description: {step.get('element_description', 'N/A')}\n"
        f"- Target Description (if applicable): {step.get('target_description', 'N/A')}\n"
        f"- Text (if applicable): {step.get('text', 'N/A')}\n\n"
        "Your task is to:\n"
        "1. Indicate if the action was successful ('success' or 'failure').\n"
        "2. If the action failed, describe the current state of the screen in a brief paragraph..\n"
        "Output format:\n"
        "{\n"
        "  \"status\": \"<success or failure>\",\n"
        "  \"advice\": \"<specific advice if failure>\"\n"
        "}"
    )

    base64_image = image_to_base64(screenshot_path)
    if not base64_image:
        return "Unable to encode image for vision advice."

    payload = {
        "model": "minicpm-v",
        "prompt": prompt,
        "max_tokens": 256,  # Ensure sufficient tokens for concise output
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

        # Extract the feedback in the expected format
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

    Args:
        template_path (str): Path to the template image.
        confidence (float): Confidence level for matching (0 to 1).

    Returns:
        dict or None: {'x': x, 'y': y} if found, else None.
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

def generate_alternative_description(element_description, context_info):
    """
    Generate an alternative element description using the LLM.

    Args:
        element_description (str): The original element description.
        context_info (str): Context information about the current screen.

    Returns:
        str or None: An alternative element description or None if generation fails.
    """
    prompt = (
        f"The automation script failed to locate the UI element with the description '{element_description}'.\n"
        f"Context: {context_info}\n"
        "Provide an alternative description for the UI element to improve detection accuracy."
    )

    payload = {
        "model": "qwen2.5-coder:14b",
        "prompt": prompt,
        "temperature": 0.4
    }

    try:
        response = requests.post(QWEN_API_URL, json=payload, headers=HEADERS, stream=True)
        response.raise_for_status()
        json_fragments = []
        for line in response.iter_lines():
            if line:
                try:
                    fragment = json.loads(line.decode("utf-8").strip())
                    if "response" in fragment:
                        json_fragments.append(fragment["response"])
                except json.JSONDecodeError:
                    logging.warning(f"Skipped non-JSON fragment while generating description: {line.decode('utf-8')}")
                    print(f"Skipped non-JSON fragment while generating description: {line.decode('utf-8')}")
    
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

def smooth_move_to(x, y):
    """Smoothly move the cursor to the specified coordinates."""
    duration = 0.5  # Adjust this value to make the movement slower or faster
    pyautogui.moveTo(x, y, duration=duration)

def execute_action(context, step, screenshot_path, retry_count=3):
    """
    Execute a UI action based on the provided step.

    Args:
        context (dict): A dictionary containing 'objective', 'completed_steps', and 'skipped_steps'.
        step (dict): A dictionary containing action details.
        screenshot_path (str): Path to the screenshot image for element detection.
        retry_count (int): Number of attempts to locate the UI element.

    Returns:
        bool: True if the action was executed successfully, False otherwise.
    """
    action = step["action"].lower()

    # Handle special actions separately
    if action == "start_browsing_mode":
        url = step.get("text", "").strip()
        if not url:
            logging.error("No URL provided for start_browsing_mode action.")
            print("Error: No URL provided for start_browsing_mode action.")
            return False  # Indicate failure

        success = start_browsing_mode(url)
        if success:
            logging.info(f"Browsing mode started with URL: {url}")
            print(f"Browsing mode started with URL: {url}")
            # Mark the step as completed
            context['completed_steps'].append(step)
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
            # Mark the step as completed
            context['completed_steps'].append(step)
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
            break  # Element found, exit the retry loop
        else:
            logging.warning(f"Attempt {attempt}/{retry_count}: Failed to locate element '{step['element_description']}'.")
            print(f"Attempt {attempt}/{retry_count}: Failed to locate element '{step['element_description']}'.")
            if attempt == retry_count:
                # Fallback to template matching
                template_path = f"templates/{step['element_description'].replace(' ', '_')}_template.png"
                coords = locate_element_with_template(template_path)
                if coords:
                    logging.info(f"Located '{step['element_description']}' using template matching.")
                    print(f"Located '{step['element_description']}' using template matching.")
                    break
                else:
                    logging.error(f"Failed to locate element '{step['element_description']}' using both PTA-1 and template matching.")
                    print(f"Failed to locate element '{step['element_description']}' using both PTA-1 and template matching.")
                    # Generate an alternative description using LLM
                    context_info = "Unable to detect the element with the current description."
                    alternative_description = generate_alternative_description(step["element_description"], context_info)
                    if alternative_description:
                        step["element_description"] = alternative_description
                        logging.info(f"Retrying with alternative description: {alternative_description}")
                        print(f"Retrying with alternative description: {alternative_description}")
                        # Retry detection once with the new description
                        coords = detect_coordinates(step["element_description"], screenshot_path)
                        if coords:
                            x, y = coords["x"], coords["y"]
                            break
                        else:
                            # Attempt template matching again with the new description
                            template_path = f"templates/{step['element_description'].replace(' ', '_')}_template.png"
                            coords = locate_element_with_template(template_path)
                            if coords:
                                logging.info(f"Located '{step['element_description']}' using template matching with alternative description.")
                                print(f"Located '{step['element_description']}' using template matching with alternative description.")
                                break
                    # If still not found, return failure
                    logging.error(f"Failed to locate element '{step['element_description']}' after generating alternative description.")
                    print(f"Failed to locate element '{step['element_description']}' after generating alternative description.")
                    return False  # All attempts failed
        sleep(2)  # Wait before retrying

    # Execute the action based on its type
    try:
        if action == "click":
            pyautogui.click(x, y)
            logging.info(f"Clicked on '{step['element_description']}' at ({x}, {y}).")
            print(f"Clicked on '{step['element_description']}' at ({x}, {y}).")
        
        elif action == "doubleclick":
            pyautogui.doubleClick(x, y)
            logging.info(f"Double-clicked on '{step['element_description']}' at ({x}, {y}).")
            print(f"Double-clicked on '{step['element_description']}' at ({x}, {y}).")
        
        elif action == "rightclick":
            pyautogui.rightClick(x, y)
            logging.info(f"Right-clicked on '{step['element_description']}' at ({x}, {y}).")
            print(f"Right-clicked on '{step['element_description']}' at ({x}, {y}).")
        
        elif action == "drag":
            target_description = step.get("target_description", "").strip()
            if not target_description:
                logging.error("The 'drag' action requires a 'target_description'.")
                print("Error: The 'drag' action requires a 'target_description'.")
                return False

            # Locate the target element
            target_coords = detect_coordinates(target_description, screenshot_path)
            if not target_coords:
                # Fallback to template matching for target
                template_path = f"templates/{target_description.replace(' ', '_')}_template.png"
                target_coords = locate_element_with_template(template_path)
                if not target_coords:
                    logging.error(f"Failed to locate target element '{target_description}' for drag action.")
                    print(f"Failed to locate target element '{target_description}' for drag action.")
                    return False
            
            target_x, target_y = target_coords["x"], target_coords["y"]

            # Perform the drag action
            pyautogui.dragTo(target_x, target_y, duration=0.5, button='left')
            logging.info(f"Dragged from '{step['element_description']}' at ({x}, {y}) to '{target_description}' at ({target_x}, {target_y}).")
            print(f"Dragged from '{step['element_description']}' at ({x}, {y}) to '{target_description}' at ({target_x}, {target_y}).")
        
        elif action == "hover":
            pyautogui.moveTo(x, y, duration=0.5)
            logging.info(f"Hovered over '{step['element_description']}' at ({x}, {y}).")
            print(f"Hovered over '{step['element_description']}' at ({x}, {y}).")
        
        elif action == "move":
            pyautogui.moveTo(x, y, duration=0.5)
            logging.info(f"Moved cursor to '{step['element_description']}' at ({x}, {y}) without clicking.")
            print(f"Moved cursor to '{step['element_description']}' at ({x}, {y}) without clicking.")
        
        elif action in ["type", "enter text into input field"]:
            text = step.get("text", "")
            if not text:
                logging.error(f"No text provided for '{action}' action.")
                print(f"Error: No text provided for '{action}' action.")
                return False
            
            pyautogui.click(x, y)  # Focus on the input field
            pyautogui.write(text, interval=0.05)
            logging.info(f"Typed '{text}' into '{step['element_description']}' at ({x}, {y}).")
            print(f"Typed '{text}' into '{step['element_description']}' at ({x}, {y}).")
        
        else:
            logging.error(f"Unknown action '{action}'.")
            print(f"Error: Unknown action '{action}'.")
            return False

    except Exception as e:
        logging.error(f"Exception occurred while executing action '{action}': {e}")
        print(f"Exception occurred while executing action '{action}': {e}")
        return False

    # After successful action execution, capture vision feedback if required
    if action not in ["type", "enter text into input field"]:
        # Only some actions might require vision feedback; adjust as needed
        pass  # Currently handled in main loop

    # For actions that don't require vision feedback, mark as completed here
    if action in ["click", "doubleclick", "rightclick", "drag", "hover", "move", "type", "enter text into input field"]:
        # Capture pre-action screenshot if available
        if screenshot_path:
            pre_action_screenshot = "screenshot_before_action.jpg"
            success_capture_before = capture_entire_screen_with_blackout(pre_action_screenshot)
            if success_capture_before:
                # Capture post-action screenshot
                post_action_screenshot = "screenshot_after_action.jpg"
                success_capture_after = capture_entire_screen_with_blackout(post_action_screenshot)
                if success_capture_after:
                    # Check action success
                    vision_result = check_action_success(pre_action_screenshot, post_action_screenshot, step["action"], step["element_description"])
                    print(f"Vision Feedback: {vision_result}")
                    logging.debug(f"Vision Feedback: {vision_result}")

                    if vision_result["status"] == "success":
                        # Mark the step as completed
                        context['completed_steps'].append(step)
                        logging.info(f"Step marked as completed based on vision feedback: {step['action']} on '{step['element_description']}'")
                        print(f"Step marked as completed based on vision feedback: {step['action']} on '{step['element_description']}'")
                    elif vision_result["status"] == "failure":
                        # Handle failure feedback
                        logging.warning("Feedback indicates a failure. Consulting the vision model for advice...")
                        print("Feedback indicates a failure. Consulting the vision model for advice...")

                        # Get advice from the vision model to adjust the strategy
                        advice = vision_result.get("advice", "No advice provided.")
                        print(f"Vision Model Advice: {advice}")
                        logging.debug(f"Vision Model Advice: {advice}")

                        if advice != "No advice provided.":
                            # Update the objective with the advice to guide future steps
                            context['objective'] = f"{context['objective']}\nNote: {advice}"
                        else:
                            # If no advice provided, optionally log or handle accordingly
                            pass
                    else:
                        logging.error("Encountered an error with vision feedback.")
                        print("Encountered an error with vision feedback.")
        else:
            # For actions without screenshots, mark as completed
            context['completed_steps'].append(step)

    return True  # Indicate successful execution

def capture_entire_screen_with_blackout(screenshot_path):
    """
    Capture the entire screen and blackout any Anaconda windows present.
    
    Args:
        screenshot_path (str): Path where the screenshot will be saved.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Capture the entire screen
        screenshot = pyautogui.screenshot().convert("RGB")
        logging.info("Captured the entire screen for screenshot.")
        print("Captured the entire screen for screenshot.")
        
        # Initialize Desktop for window management
        desktop = Desktop(backend="win32")
        
        # Iterate through all open windows to find Anaconda windows
        for window in desktop.windows():
            if "Anaconda" in window.window_text():
                rect = window.rectangle()
                logging.info(f"Found Anaconda window: {window.window_text()} at {rect}")
                print(f"Found Anaconda window: {window.window_text()} at {rect}")
                
                # Draw a black rectangle over the Anaconda window region
                draw = ImageDraw.Draw(screenshot)
                draw.rectangle([rect.left, rect.top, rect.right, rect.bottom], fill="black")
                logging.info(f"Blacked out Anaconda window at ({rect.left}, {rect.top}, {rect.right}, {rect.bottom}).")
                print(f"Blacked out Anaconda window at ({rect.left}, {rect.top}, {rect.right}, {rect.bottom}).")
        
        # Save the modified screenshot
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
        'use_vision': True  # Will be updated based on user input
    }

    failed_attempts = 0
    max_failed_attempts = 25  # Increased to allow more retries
    repeated_steps = 0
    max_repeated_steps = 25  # Increased to allow more repeated steps

    # Configuration: Define the delay duration after executing an action (in seconds)
    post_action_delay = 1  # Adjust this value as needed (e.g., 0.5 seconds)

    # Prompt user for vision model activation and objective
    vision_enabled_input = input("Enable vision model? (yes/no): ").strip().lower()
    context['use_vision'] = vision_enabled_input == "yes"
    context['objective'] = input("Enter the objective: ")

    # Main automation loop
    while failed_attempts < max_failed_attempts and repeated_steps < max_repeated_steps:
        try:
            print("\nFetching the next step...")
            logging.info("\nFetching the next step...")

            # Fetch the next step from the language model
            steps = get_next_step(context)

            if not steps:
                logging.info("No steps generated. Exiting.")
                print("No steps generated. Exiting.")
                break  # Exit if no steps are returned

            step = steps[0]  # Consider only the first step

            # Generate a unique identifier for the step to track completion
            if step["action"].lower() == "start_browsing_mode":
                step_id = f"{step['action'].lower()}_{step['text'].lower()}"
            elif step["action"].lower() == "stop_browsing_mode":
                step_id = f"{step['action'].lower()}"
            else:
                step_id = f"{step['action'].lower()}_{step['element_description'].lower()}"

            # Check if the step has already been completed
            if any(
                (completed_step["action"].lower() == step["action"].lower() and
                 completed_step.get("text", "").lower() == step.get("text", "").lower()) if "text" in step else
                (completed_step["action"].lower() == step["action"].lower() and
                 completed_step.get("element_description", "").lower() == step.get("element_description", "").lower())
                for completed_step in context['completed_steps']
            ):
                context['skipped_steps'].append(step)
                repeated_steps += 1
                logging.info(f"Step already completed successfully. Skipping to the next step. (Repeated {repeated_steps}/{max_repeated_steps})")
                print(f"Step already completed successfully. Skipping to the next step. (Repeated {repeated_steps}/{max_repeated_steps})")
                continue  # Skip to the next iteration
            else:
                repeated_steps = 0  # Reset repeated_steps counter

            print(f"Processing step: {step}")
            logging.info(f"Processing step: {step}")

            # Determine if action requires detection
            if step["action"].lower() not in ["start_browsing_mode", "stop_browsing_mode"]:
                # Capture pre-action screenshot with blackout
                pre_action_screenshot = "screenshot_before_action.jpg"
                success_capture_before = capture_entire_screen_with_blackout(pre_action_screenshot)
                if not success_capture_before:
                    failed_attempts += 1
                    logging.error("Pre-action screenshot capture failed.")
                    print("Error: Pre-action screenshot capture failed.")
                    continue  # Skip to the next iteration if screenshot capture failed

                # Execute action with pre-action screenshot
                success = execute_action(context, step, pre_action_screenshot, retry_count=3)
            else:
                # For 'start_browsing_mode' and 'stop_browsing_mode', no detection needed
                success = execute_action(context, step, None, retry_count=3)

            if success:
                logging.info(f"Step '{step['action']}' on '{step.get('element_description', '')}' executed.")
                print(f"Step '{step['action']}' on '{step.get('element_description', '')}' executed.")
            else:
                failed_attempts += 1
                logging.error(f"Action execution failed: {step['action']} on '{step.get('element_description', '')}' (Attempt {failed_attempts}/{max_failed_attempts})")
                print(f"Action execution failed: {step['action']} on '{step.get('element_description', '')}' (Attempt {failed_attempts}/{max_failed_attempts})")
                continue  # Skip vision processing if action failed to execute

            # If vision is enabled and the action requires feedback
            if context['use_vision'] and step["action"].lower() not in ["start_browsing_mode", "stop_browsing_mode"]:
                # Introduce a slight delay to allow the UI to update after the action
                sleep(post_action_delay)
                logging.debug(f"Waiting for {post_action_delay} seconds to allow UI to update.")
                print(f"Waiting for {post_action_delay} seconds to allow UI to update.")

            # Brief pause before the next iteration
            sleep(1)

        except Exception as e:
            # Handle any unexpected exceptions gracefully
            logging.error(f"An error occurred: {e}")
            print(f"Error: An error occurred: {e}")
            failed_attempts += 1

    # Log and notify upon completion or reaching maximum failed attempts
    logging.info("\nAutomation process completed or reached maximum failed attempts.")
    print("\nAutomation process completed or reached maximum failed attempts.")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
