# Automation Assistant for UI Task Execution.

This project is a Python-based automation assistant designed for task execution within a Windows environments. It utilizes advanced language models and vision capabilities to interact with user interfaces, attempting to automate repetitive tasks.

## Key Features

### 1. **Model Integration**
- **Language Models**: Includes support for advanced models like `qwen2.5-coder` and `minicpm-v` for generating and verifying tasks.
- **Vision Models**: Uses `AskUI/PTA-1` for object detection.

### 2. **UI Automation**
- Automates actions like:
  - Clicking
  - Typing
  - Drag-and-drop
  - Starting and stopping browsing sessions
  - Hovering over elements
- Supports dynamic detection of UI elements using descriptions and screenshots.

### 3. **Vision Feedback**
- Incorporates pre- and post-action screenshots to verify task execution success.
- Provides actionable feedback in case of failures to retry or adjust the task.

### 4. **Browsing Mode**
- Integrates browsing automation for navigating to specific URLs.
- Supports conditional actions based on browsing state.

### 5. **Error Handling and Logging**
- Comprehensive logging to debug issues effectively.
- Handles retries and alternative descriptions for failed detections.

## Project Structure

```plaintext
├── main.py                 # Main script for the automation process
├── templates/              # Contains templates for UI element matching
├── requirements.txt        # Python dependencies
├── automation_debug.log    # Log file for debugging
├── screenshots/            # Folder to store pre- and post-action screenshots
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/automation-assistant.git
   cd automation-assistant
   ```

2. Create a virtual environment and activate it:
   ```bash
   conda create -n automation-assistant python=3.8 -y
   conda activate automation-assistant
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Install additional tools like Google Chrome if required for browsing automation.

## Usage

1. Configure the logging settings in `main.py` if needed.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Follow the prompts to specify the objective and enable or disable vision model integration.

## Dependencies

- Python 3.8+
- `torch`
- `transformers`
- `Pillow`
- `pyautogui`
- `requests`
- `pywinauto`

Install all dependencies via the `requirements.txt` file.

## Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- [AskUI](https://askui.com/) for the PTA-1 model
