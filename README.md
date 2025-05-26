# 🧠 LLM Evaluation Dashboard Agent

This is an agentic, graph-based framework for evaluating the outputs of Large Language Models (LLMs) using hallucination detection techniques.

Built using [LangGraph](https://github.com/langchain-ai/langgraph), the project orchestrates a flow of evaluation agents that process prompts, generate responses, assess hallucination risk, and log results in a configurable, scalable pipeline.

---

## 🚀 Features

- 🧱 Agent-based modular design using LangGraph
- ✅ Multiple hallucination detection methods (LLM-as-a-Judge first, more coming)
- ⚙️ Configurable via `config.yaml`
- 🧪 Load test prompts and ground truths from JSON files
- 📊 Tracks tokens, latency, model version, and evaluation scores

---
## 🧪 Example Test Case

```json
{
  "prompt": "What is the difference between synchronous and asynchronous programming in Python?",
  "ground_truth": "...",
  "model": "gpt-4o",
  "metadata": {
    "use_case": "technical explanation",
    "ground_truth_type": "text"
  }
}
```

---

## 🔮 Roadmap

### ✅ Completed
- [x] LangGraph setup with typed shared state (`EvalState`)
- [x] Input handler agent to prepare prompt state
- [x] Model runner agent to call OpenAI (GPT-4o)
- [x] Config-driven architecture via `config.yaml`
- [x] Hallucination detection via LLM-as-a-Judge (OpenAI)
- [x] JSON prompt ingestion for modular test cases

### 🔜 In Progress / Planned
- [ ] Add hallucination detection via embedding similarity
- [ ] Add Vectara hallucination evaluation model (Hugging Face)
- [ ] Telemetry logging agent (latency, token usage, verdicts)
- [ ] Evaluation metrics agent (tone, coherence, completeness scoring)
- [ ] CLI or batch runner for test prompt files
- [ ] Result logger (to JSONL or CSV)
- [ ] Dashboard or Streamlit front-end for visualizing evaluations