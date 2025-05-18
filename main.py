import json
import logging
from langgraph.graph import StateGraph, END
from agents.input_handler import input_handler_node
from agents.model_runner import model_runner_node
from agents.hallucination_checker import hallucination_checker_node
from state.graph_schema import EvalState
from config.settings import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# 1. Define a graph builder
builder = StateGraph(EvalState)

# 2. Register nodes
builder.add_node("input_handler", input_handler_node)
builder.add_node("model_runner", model_runner_node)
builder.add_node("hallucination_checker", hallucination_checker_node)

# 3. Define edges between nodes
builder.set_entry_point("input_handler")
builder.add_edge("input_handler", "model_runner")
builder.add_edge("model_runner", "hallucination_checker")
builder.add_edge("hallucination_checker", END)

# 4. Compile the graph
graph = builder.compile()

# 5. Run a test
if __name__ == "__main__":
    with open("data/prompts/sync_vs_async.json", "r") as f:
        test_input = json.load(f)

    logger.info("Starting graph execution with test input: %s", test_input)
    final_state = graph.invoke(test_input)

    logger.info("Graph execution completed. Final state: %s", final_state)

    print("\n=== Final Output ===")
    print("🧠 Response:\n", final_state.get("response"))
    print("\n📊 Hallucination Detection:")
    print(" - Verdict:", final_state.get("hallucination_verdict"))
    print(" - Score:", final_state.get("hallucination_score"))
    print(" - Debug Info:", final_state.get("hallucination_debug"))

    print("\n⚙️ Meta:")
    print(" - Model:", final_state.get("model_version"))
    print(" - Latency (ms):", final_state.get("latency_ms"))
    print(" - Tokens used:", final_state.get("tokens_used"))