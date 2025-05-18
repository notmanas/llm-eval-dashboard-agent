import logging
import time
import openai
from typing import Dict
from state.graph_schema import EvalState
from config.settings import get_openai_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def model_runner_node(state: EvalState) -> EvalState:
    """
    Sends prompt to OpenAI model and returns the response with telemetry.
    """
    logger.info("Model runner node invoked with state: %s", state)

    client = get_openai_client()
    prompt = state["prompt"]
    model = state.get("model", "gpt-4o")

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
    except Exception as e:
        logger.error("Error while calling OpenAI API: %s", e)
        raise

    end_time = time.time()

    message = response.choices[0].message.content
    latency_ms = (end_time - start_time) * 1000
    tokens_used = response.usage.total_tokens if response.usage else -1
    model_version = response.model

    updated_state: EvalState = {
        **state,
        "response": message,
        "latency_ms": round(latency_ms, 2),
        "tokens_used": tokens_used,
        "model_version": model_version
    }

    logger.info("Model runner node updated state: %s", updated_state)
    return updated_state
