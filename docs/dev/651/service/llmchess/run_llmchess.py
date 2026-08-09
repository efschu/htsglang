"""Driver for the llm_chess benchmark against the local OpenAI-compatible endpoint.

Board config comes from /opt/llmchess/llm_chess/.env (MODEL_KIND_B=local, ...).
White is the random-mover, Black is the local LLM.

Env knobs:
  LLMCHESS_MAX_MOVES  maximum game moves before termination (default 200)
  LLMCHESS_LOG_DIR    directory for the game log (default ~/llmchess-runs)
  LLMCHESS_API_RETRIES  retries per API error (default 6; the endpoint is
                        on-demand and a cold reload takes ~150 s)
  LLMCHESS_RETRY_DELAY  base seconds between retries, exponential (default 20)
  LLMCHESS_THINKING     "1" to leave the model's thinking mode on (default off)
  LLMCHESS_MAX_TOKENS   per-reply completion cap (default 512)
"""
import json
import os
import sys

APP = "/opt/llmchess/llm_chess"
os.chdir(APP)
sys.path.insert(0, APP)

import llm_chess  # noqa: E402
from utils import get_llms  # noqa: E402

llm_chess.max_game_moves = int(os.environ.get("LLMCHESS_MAX_MOVES", "200"))
llm_chess.dialog_turn_delay = float(os.environ.get("LLMCHESS_TURN_DELAY", "0"))
llm_chess.throttle_delay = 0
llm_chess.visualize_board = False
llm_chess.random_print_board = False
# The endpoint is on-demand: a parked model needs ~150 s to reload, so a single
# request can fail while the backend restarts. Retry long enough to ride that out.
llm_chess.max_api_retries = int(os.environ.get("LLMCHESS_API_RETRIES", "6"))
llm_chess.api_retry_delay = float(os.environ.get("LLMCHESS_RETRY_DELAY", "20"))

log_dir = os.environ.get("LLMCHESS_LOG_DIR") or os.path.join(
    os.path.expanduser("~"), "llmchess-runs"
)
os.makedirs(log_dir, exist_ok=True)

# The local backend's KV pool is sized per load and can be as small as a few
# thousand tokens. With thinking mode on, one runaway <think> block both
# exceeds that window and — when it is truncated before its closing tag —
# survives the history filter, so every later turn carries it and the context
# overflows for good. Thinking is therefore off by default here, and each reply
# is capped, which keeps a whole game inside even the smallest observed pool.
thinking_on = os.environ.get("LLMCHESS_THINKING", "0") == "1"
max_tokens = int(os.environ.get("LLMCHESS_MAX_TOKENS", "512"))

provider_overrides = {"max_tokens": max_tokens}
if not thinking_on:
    provider_overrides["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": False}
    }

model_config = {
    "hyperparams": llm_chess.default_hyperparams,
    "provider_overrides": provider_overrides,
}
llm_config_white, llm_config_black = get_llms(
    white_hyperparams=dict(model_config),
    black_hyperparams=dict(model_config),
)

print(
    "CONFIG base_url=%s model=%s max_game_moves=%d log_dir=%s thinking=%s max_tokens=%d"
    % (
        os.environ.get("LOCAL_BASE_URL_B"),
        os.environ.get("LOCAL_MODEL_NAME_B"),
        llm_chess.max_game_moves,
        log_dir,
        "on" if thinking_on else "off",
        max_tokens,
    ),
    flush=True,
)

stats, white, black = llm_chess.run(
    log_dir=log_dir,
    llm_config_white=llm_config_white,
    llm_config_black=llm_config_black,
)
print("LLMCHESS_STATS " + json.dumps(stats, default=str), flush=True)
print("LLMCHESS_DONE_OK", flush=True)
