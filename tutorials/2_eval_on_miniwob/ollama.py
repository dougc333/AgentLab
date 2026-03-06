import re
# import litellm  # Ensure litellm is imported in your actual script

# 1. PLACE AT THE TOP: Define the System Prompt
# This sets the "rules of the world" for the AI.
SYSTEM_PROMPT = """You are a web-agent named GenericAgent. Your goal is to solve tasks on a website using the tools provided.

# Rules for Interaction:
1. PERCEPTION: You will receive an Accessibility Tree (AXTree) where elements have a 'bid' (e.g. [20]).
2. REASONING: You must think step-by-step before acting. Wrap your thoughts in <think></think> tags.
3. ACTION: You must provide EXACTLY ONE action per turn. Wrap it in <action></action> tags.
4. FORMAT: Do not repeat the action outside of the tags. Do not use Markdown code blocks (```) inside tags.

# Action Space:
- click(bid: str): Clicks the element with the given bid.
- fill(bid: str, value: str): Types text into the element.
- type(bid: str, value: str, press_enter: bool = True): Types and optionally presses enter.
- scroll(x: float, y: float): Scrolls the page.
- wait(ms: float): Pauses execution.
- mouse_click(x: float, y: float): Clicks specific coordinates.

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
    Utility to find the first <action> tag and return its content. 
    This fixes the 'double action' bug where the model repeats itself.
    """
    match = re.search(r'<action>(.*?)</action>', response_text, re.DOTALL)
    if match:
        action = match.group(1).strip()
        # Remove markdown backticks if the model ignores the prompt rules
        action = re.sub(r'^```[a-z]*\n?', '', action)
        action = re.sub(r'\n?```$', '', action)
        return action.strip()
    
    # Simple fallback if tags are missing
    fallback = re.search(r'(\w+\([\'"].*?[\'"]\))', response_text)
    return fallback.group(1) if fallback else "noop()"

# 2. PLACE INSIDE THE FUNCTION: Inject the prompt into messages
def get_agent_response(model, messages):
    """
    Handles the communication with Ollama via LiteLLM.
    """
    # Ensure our rigid system prompt is the VERY FIRST message
    if not messages or messages[0].get('role') != 'system':
        messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    
    # Example LiteLLM call (uncomment/adjust for your actual environment)
    # response = litellm.completion(
    #     model=model, 
    #     messages=messages,
    #     temperature=0.2  # Keep temp low for consistency
    # )
    
    # raw_content = response.choices[0].message.content
    # return extract_action(raw_content)
    
    return "Action extracted successfully"



import os, sys, logging
import requests
import litellm
from dataclasses import dataclass, asdict
from copy import deepcopy

# 1. Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

# 2. Imports for AgentLab
from bgym import DEFAULT_BENCHMARKS
from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS, GenericAgentArgs
from agentlab.experiments.study import Study

# 3. The Custom Bridge Class (Now as a Dataclass)
@dataclass
class CustomOllamaArgs:
    model_name: str = "ollama/qwen3-vl:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_new_tokens: int = 512
    max_total_tokens: int = 32768
    max_input_tokens: int = 32000
    vision_support: bool = True
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0

    def __post_init__(self):
        # Unique ID for Pandas grouping/sorting
        self.id = f"{self.model_name}_{self.temperature}"

    # Pandas sorting support
    def __lt__(self, other):
        return self.id < other.id
    
    def __eq__(self, other):
        return isinstance(other, CustomOllamaArgs) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def prepare_server(self):
        return True

    def close_server(self):
        return True

    def make_model(self):
        """
        Directly wraps LiteLLM to bypass internal AgentLab registry issues.
        Satisfies methods like get_stats() and returns structured dicts.
        """
        class SimpleModel:
            def __init__(self, m, b, t):
                self.model_name = m
                self.api_base = b
                self.temperature = t
                self.stats = {
                    "prompt_tokens": 0, 
                    "completion_tokens": 0, 
                    "total_cost": 0.0
                }
            
            def get_stats(self):
                return self.stats

            def __call__(self, messages, **kwargs):
                # Clean kwargs to prevent duplicate 'model' argument
                kwargs.pop('model', None)
                
                response = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    api_base=self.api_base,
                    temperature=self.temperature,
                    **kwargs
                )
                
                # Update usage statistics
                usage = getattr(response, "usage", None)
                if usage:
                    self.stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
                    self.stats["completion_tokens"] += getattr(usage, "completion_tokens", 0)
                
                content = response.choices[0].message.content
                
                # Return dictionary with 'role' as expected by BaseMessage constructor
                return {
                    "role": "assistant",
                    "content": content
                }
        
        return SimpleModel(self.model_name, self.base_url, self.temperature)

# 4. Configuration and Benchmark Setup
ollama_args = CustomOllamaArgs()
agent_config = GenericAgentArgs(chat_model_args=ollama_args, flags=BASE_FLAGS)

# miniwob_tiny_test includes tasks like 'click-dialog' and 'click-checkboxes'
benchmark = DEFAULT_BENCHMARKS["miniwob_tiny_test"]()

if __name__ == "__main__":
    # n_jobs=1 is critical for local LLMs/VLMs in Colab
    study = Study([agent_config], benchmark)
    study.run(n_jobs=1, parallel_backend="joblib")