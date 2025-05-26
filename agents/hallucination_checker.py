# agents/hallucination_checker.py

from typing import Dict
from state.graph_schema import EvalState
from config.settings import get_openai_client
import yaml
import os
import re
import json

from utils.embedding_utils import get_embedding, cosine_similarity

# Load config.yaml once
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def clean_json_output(text: str) -> str:
    """
    Remove markdown-style code blocks from LLM output before JSON parsing.
    """
    cleaned = re.sub(r"^```json\n", "", text.strip())
    cleaned = re.sub(r"^```\n?", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()


def hallucination_embedding(prompt, response, ground_truth):
    response_emb = get_embedding(response)
    truth_emb = get_embedding(ground_truth)
    sim = cosine_similarity(response_emb, truth_emb)
    threshold = 0.8
    verdict = "faithful" if sim >= threshold else "hallucinated"
    return {
        "hallucination_score": float(sim),
        "verdict": verdict,
        "details": {"cosine_similarity": float(sim)}
    }


def hallucination_vectara(prompt, response, ground_truth):
    # TODO: Implement Vectara or other method
    return {
        "hallucination_score": 0.72,  # Example score based on Vectara analysis
        "verdict": "POSSIBLE_HALLUCINATION",  # Example verdict
        "details": {}  # Example details, could include Vectara-specific info
    }


def hallucination_llm_judge(prompt: str, response: str, ground_truth: str) -> Dict:
    judge_prompt = f"""
You are a fact-checking AI assistant.

Here is the original user prompt:
{prompt}

Here is the model's response:
{response}

Here is the correct ground truth:
{ground_truth}

Please assess whether the model's response is faithful to the ground truth.
Respond in the following JSON format:
{{
  "hallucination_score": float (0.0 = completely inaccurate, 1.0 = completely accurate),
  "verdict": "faithful" or "hallucinated"
}}
"""

    client = get_openai_client()
    result = client.chat.completions.create(
        model=config["openai"]["model"],
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )

    try:
        cleaned_text = clean_json_output(result.choices[0].message.content)
        judge_output = json.loads(cleaned_text)
    except Exception as e:
        judge_output = {
            "hallucination_score": -1,
            "verdict": "error",
            "error": str(e),
            "raw_output": result.choices[0].message.content
        }

    return judge_output


def hallucination_checker_node(state: EvalState) -> EvalState:
    method = config.get("hallucination", {}).get("method", "llm_judge")

    if method == "llm_judge":
        result = hallucination_llm_judge(
            prompt=state["prompt"],
            response=state["response"],
            ground_truth=state["ground_truth"]
        )
    elif method == "embedding":
        result = hallucination_embedding(
            prompt=state["prompt"],
            response=state["response"],
            ground_truth=state["ground_truth"]
        )
    elif method == "vectara":
        result = hallucination_vectara(
            prompt=state["prompt"],
            response=state["response"],
            ground_truth=state["ground_truth"]
        )
    else:
        raise NotImplementedError(f"Hallucination method '{method}' not implemented.")

    return {
        **state,
        "hallucination_score": result.get("hallucination_score"),
        "hallucination_verdict": result.get("verdict"),
        "hallucination_debug": result  # Optional: keep raw output for debugging
    }

