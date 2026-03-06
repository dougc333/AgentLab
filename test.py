import os
from copy import deepcopy

from bgym import DEFAULT_BENCHMARKS
from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS, CHAT_MODEL_ARGS_DICT, GenericAgentArgs
from agentlab.experiments.study import Study


import litellm

litellm.register_model({
  "Qwen/Qwen2-1.5B-Instruct": {
    "litellm_provider": "openai",   # vLLM speaks OpenAI protocol
    "mode": "chat",
    "max_tokens": 2048,            # match /v1/models max_model_len
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
  }
})

# vLLM usually doesn't require a real key, but the OpenAI client requires *something*
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:8000/v1"   # for older codepaths
os.environ["OPENAI_API_KEY"] = "local"
vllm_args = deepcopy(CHAT_MODEL_ARGS_DICT["openai/gpt-3.5-turbo-1106"])

# IMPORTANT: use your vLLM-served model name here
vllm_args.model_name = "Qwen/Qwen2-1.5B-Instruct"

# IMPORTANT: vLLM OpenAI server uses /v1
vllm_args.base_url = "http://127.0.0.1:8000/v1"

vllm_args.temperature = 0.2
vllm_args.max_new_tokens = 64
vllm_args.vision_support = False

agent_config = GenericAgentArgs(chat_model_args=vllm_args, flags=BASE_FLAGS)

# Use a tiny benchmark to sanity-check quickly
benchmark = DEFAULT_BENCHMARKS["miniwob_tiny_test"]()

if __name__ == "__main__":
    Study([agent_config], benchmark).run(n_jobs=1, parallel_backend="joblib")
