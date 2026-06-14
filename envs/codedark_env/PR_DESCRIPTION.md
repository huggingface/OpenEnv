# Add CodeDark: Multi-turn Data Analytics Environment

## Summary

This PR adds **CodeDark**, the first data analytics environment for the OpenEnv Hub. CodeDark challenges AI agents to analyze real CSV datasets using Python/Pandas through multi-turn tool-use conversations, testing their ability to be data scientists rather than just code executors.

## Key Features

- **Real Business Tasks**: Bank marketing (750K rows) and road safety (500K rows) datasets with genuine analytical questions
- **Multi-Turn Interaction**: Agents explore data, save notes, ask clarifications, and submit answers over multiple turns
- **Shaped Rewards**: 80% correctness + 10% efficiency + 10% token cost for RL training
- **Pre-Benchmarked**: 25 curated L4-L6 difficulty tasks validated on 11+ models with 1,844 completions
- **Live Demo**: Already deployed at https://huggingface.co/spaces/albert-einstein-09/codedark

## Benchmark Results

Pre-benchmarked performance on the 25-task benchmark:

| Model            | Accuracy | Avg Turns | Cost/Task |
| ---------------- | -------- | --------- | --------- |
| Claude Opus 4.5  | 77.3%    | 4.2       | $0.89     |
| Qwen3 Max        | 46.7%    | 5.1       | $0.12     |
| Mistral Large    | 45.3%    | 5.8       | $0.18     |
| Llama 4 Maverick | 38.7%    | 6.2       | $0.08     |

Full leaderboard: https://www.analytics-rl.com

## Tools

| Tool            | Description                              |
| --------------- | ---------------------------------------- |
| `run_python`    | Execute Python/pandas code (sandboxed)   |
| `read_notes`    | Read saved notes from previous turns     |
| `save_note`     | Save observations for later recall       |
| `clarify`       | Ask clarifying questions (max 2/episode) |
| `submit_answer` | Submit final answer (ends episode)       |

## Task Difficulty Levels

| Level | Complexity      | Example                                         |
| ----- | --------------- | ----------------------------------------------- |
| L4    | Quartile/binned | "Subscription rate in Q1 balance?"              |
| L5    | Multi-condition | "Rate for month='may' AND job='management'?"    |
| L6    | Nested extrema  | "In lowest subscription month, what's avg day?" |

## Quick Start

```python
from codedark_env import CodeDarkEnv

# Connect to environment
env = CodeDarkEnv("https://albert-einstein-09-codedark.hf.space")

# Reset for new task
obs = env.reset()
print(f"Task: {obs['question']}")
# Task: What's the subscription rate for month='may' AND job='management' AND balance in Q1?

# Execute Python code
obs = env.run_python("result = df.shape")
print(f"Shape: {obs['stdout']}")
# Shape: run_python Result: (750000, 18)

# Submit answer
obs = env.submit_answer(2.44)
print(f"Reward: {obs['reward']}")
# Reward: 0.98
```

## Files Added

```
envs/codedark_env/
├── __init__.py           # Package exports
├── models.py             # Action, Observation, State dataclasses
├── client.py             # HTTP client implementation
├── README.md             # Full documentation
└── server/
    ├── __init__.py
    ├── environment.py    # Core environment logic
    ├── tools.py          # Tool implementations
    ├── scoring.py        # Reward computation
    ├── app.py            # FastAPI server
    ├── requirements.txt  # Python dependencies
    └── Dockerfile        # Container spec
```

## Checklist

- [x] Environment follows OpenEnv spec with `reset()`, `step()`, `state` API
- [x] Pydantic models for Action, Observation, State
- [x] FastAPI server with standard endpoints
- [x] Docker container builds and runs
- [x] README with action/observation specs
- [x] Pre-benchmarked on multiple models
- [x] Live demo on HuggingFace Spaces

## Links

- **HuggingFace Space**: https://huggingface.co/spaces/albert-einstein-09/codedark
- **Live API**: https://albert-einstein-09-codedark.hf.space
- **Leaderboard**: https://www.analytics-rl.com
- **Original Benchmark**: https://github.com/vj-09/codeblue-env

## Author

**Vijay Athithya** ([@vj-09](https://github.com/vj-09))

---

cc @jspisak
