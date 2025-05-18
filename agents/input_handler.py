import logging
from typing import Dict
from state.graph_schema import EvalState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def input_handler_node(state: EvalState) -> EvalState:
    """
    Validates and initializes the shared state.
    """
    logger.info("Input handler node invoked with state: %s", state)

    prompt = state.get("prompt")
    model = state.get("model", "gpt-4")  # Default to GPT-4
    metadata = state.get("metadata", {})

    if not prompt:
        logger.error("Prompt is missing in the input state.")
        raise ValueError("Prompt is required in input state.")

    updated_state: EvalState = {
        **state,
        "prompt": prompt.strip(),
        "model": model,
        "metadata": metadata,
    }

    logger.info("Input handler node updated state: %s", updated_state)
    return updated_state
