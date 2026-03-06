import re
import litellm
import logging

# Set up logging to monitor LiteLLM and Ollama interactions
logging.basicConfig(level=logging.INFO)

# 1. THE SYSTEM PROMPT
# This defines the "Rules of Engagement" for the model. 
# It forces the use of XML tags and prevents common formatting errors.
SYSTEM_PROMPT = """You are a web-agent named GenericAgent. Your goal is to solve tasks on a website using the tools provided.

# Rules for Interaction:
1. PERCEPTION: You will receive an Accessibility Tree (AXTree) where elements have a 'bid' (e.g. [20]).
2. REASONING: You must think step-by-step before acting. Wrap your thoughts in <think></think> tags.
3. ACTION: You must provide EXACTLY ONE action per turn. Wrap it in <action></action> tags.
4. FORMAT: Do not repeat the action outside of the tags. Do not use Markdown code blocks (```) inside tags.
5. NO HALLUCINATION: Only use the functions listed in the Action Space below.

# Action Space:
- click(bid: str): Clicks the element with the given bid identifier.
- fill(bid: str, value: str): Enters text into the input field identified by bid.
- type(bid: str, value: str, press_enter: bool = True): Types text and optionally presses Enter.
- scroll(x: float, y: float): Scrolls the page to coordinates.
- wait(ms: float): Pauses execution for a specified duration.
- mouse_click(x: float, y: float): Performs a raw mouse click at coordinates (x, y).
- noop(): Do nothing for this turn.

# Example Response:
<think>
The user wants to submit the form. I see the 'Submit' button has bid='33'.
</think>
<action>
click('33')
</action>
"""

def extract_action(response_text: str) -> str:
    """
    Finds the first <action> tag and returns its content. 
    This is the primary defense against the model repeating itself 
    or wrapping code in markdown backticks.
    """
    # 1. Look for content inside <action>...</action>
    match = re.search(r'<action>(.*?)</action>', response_text, re.DOTALL)
    if match:
        action = match.group(1).strip()
        # Clean up any markdown code blocks the model might have added
        action = re.sub(r'^```[a-z]*\n?', '', action)
        action = re.sub(r'\n?```$', '', action)
        return action.strip()
    
    # 2. Fallback: If tags are missing, search for a valid function call pattern
    fallback = re.search(r'(\w+\([\'"].*?[\'"]\))', response_text)
    if fallback:
        return fallback.group(1)
        
    return "noop()"

def get_agent_response(model_name, messages, max_retries=3):
    """
    Communicates with the Ollama model via LiteLLM.
    Ensures the system prompt is injected and actions are cleaned.
    """
    # Inject the System Prompt as the first message if it isn't already there
    if not messages or messages[0].get('role') != 'system':
        messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})

    for attempt in range(max_retries):
        try:
            # We use a low temperature for consistency in action generation
            response = litellm.completion(
                model=model_name,
                messages=messages,
                temperature=0.1,  # Lowered for even stricter output
                num_retries=2
            )
            
            raw_content = response.choices[0].message.content
            # Clean the response to get just the executable code
            action = extract_action(raw_content)
            
            # Log the extracted action for debugging in the console
            print(f"--- Action Processed: {action} ---")
            return action

        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logging.error("All retries exhausted for Ollama model.")
                return "noop()"

# Boilerplate for running a standalone test of the agent
if __name__ == "__main__":
    test_messages = [
        {"role": "user", "content": "AXTree: [10] checkbox 'Accept Terms'. Task: Accept terms and submit."}
    ]
    # Use the model string from your evaluation logs
    result = get_agent_response("ollama/qwen3-vl:latest", test_messages)
    print(f"Final Action: {result}")
