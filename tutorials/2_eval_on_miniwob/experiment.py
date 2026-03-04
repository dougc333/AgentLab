import os, sys, logging
import litellm
from copy import deepcopy

# 1. Handle API Keys
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set. Run: export HF_TOKEN='hf_...'")

# Map your HF token to the OpenAI key variable LiteLLM expects for the 'openai' provider
os.environ["OPENAI_API_KEY"] = HF_TOKEN

# 2. Register the models in LiteLLM
# This bypasses the "Model isn't mapped" error while keeping the model ID clean for the HF Router
litellm.register_model({
    "Qwen/Qwen2.5-7B-Instruct": {
        "max_tokens": 32768,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openai",  # Force the OpenAI protocol
        "mode": "chat"
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "max_tokens": 131072,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openai",
        "mode": "chat"
    }
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

from bgym import DEFAULT_BENCHMARKS
from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS, CHAT_MODEL_ARGS_DICT, GenericAgentArgs
from agentlab.experiments.study import Study

# 3. Configure Model Args
hf_args = deepcopy(CHAT_MODEL_ARGS_DICT["openai/gpt-5-nano-2025-08-07"])

# Use the clean ID that worked in your curl
hf_args.model_name = "Qwen/Qwen2.5-7B-Instruct" 

# Set the base_url to the HF Router (Note: use 'base_url', not 'api_base' for LiteLLMModelArgs)
hf_args.base_url = "https://router.huggingface.co/v1"

hf_args.temperature = 0.2
hf_args.max_new_tokens = 512
hf_args.vision_support = False

agent_config = GenericAgentArgs(chat_model_args=hf_args, flags=BASE_FLAGS)
benchmark = DEFAULT_BENCHMARKS["miniwob"]()

if __name__ == "__main__":
    # n_jobs=1 is safer for initial testing to see logs clearly
    Study([agent_config], benchmark).run(n_jobs=1, parallel_backend="joblib")
