# CodeDark Environment

**Multi-turn data analytics environment for training RL agents on real business tasks.**

CodeDark challenges AI agents to analyze CSV datasets using Python/Pandas, testing their ability to be data scientists rather than just code executors. It's the first data analytics environment in the OpenEnv ecosystem.

## Overview

- **Task Type**: Multi-turn data analytics with pandas/numpy
- **Datasets**: Bank Marketing (750K rows) + Road Safety (500K rows)
- **Benchmark**: 25 curated L4-L6 difficulty tasks
- **Reward**: Shaped (80% correctness + 10% efficiency + 10% token cost)
- **Pre-validated**: 11+ models evaluated, 1,844 completions

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

# Explore data
obs = env.run_python("result = df.columns.tolist()")

# Calculate answer
obs = env.run_python("""
df['_q'] = pd.qcut(df['balance'], 4, labels=['Q1','Q2','Q3','Q4'])
filtered = df[(df['month'] == 'may') & (df['job'] == 'management') & (df['_q'] == 'Q1')]
result = round((filtered['y'] == 1).mean() * 100, 2)
""")

# Submit answer
obs = env.submit_answer(2.44)
print(f"Reward: {obs['reward']}")
# Reward: 0.98
```

## Action Space

Actions are JSON objects with `tool` and `args` fields:

```python
class CodeDarkAction:
    tool: Literal["run_python", "read_notes", "save_note", "clarify", "submit_answer"]
    args: str
```

### Tools

| Tool            | Args Format                          | Description                                                    |
| --------------- | ------------------------------------ | -------------------------------------------------------------- |
| `run_python`    | `<code>python_code</code>`           | Execute Python/pandas code. Store result in `result` variable. |
| `read_notes`    | (empty)                              | Read all saved notes from previous turns.                      |
| `save_note`     | `note_content`                       | Save observation for later recall. Notes persist across turns. |
| `clarify`       | `<question>your question</question>` | Ask clarifying question (max 2 per episode).                   |
| `submit_answer` | `<answer>value</answer>`             | Submit final answer. Ends episode.                             |

## Observation Space

```python
class CodeDarkObservation:
    stdout: str           # Tool output
    stderr: str           # Error output
    exit_code: int        # 0 = success, 1 = error
    turn: int             # Current turn (1 to max_turns)
    max_turns: int        # Maximum turns (default: 10)
    notes: List[str]      # Saved notes
    task_id: str          # Task identifier
    question: str         # Task question
    difficulty: str       # L4, L5, or L6
    dataset: str          # "bank" or "road"
    done: bool            # Episode complete?
    submitted: bool       # Answer submitted?
    reward: Optional[float]      # Total reward (only when done)
    correctness: Optional[float] # Correctness component
    efficiency: Optional[float]  # Efficiency component
```

## Reward Structure

Total reward (max 1.0) computed from three components:

| Component   | Weight | Description                                                       |
| ----------- | ------ | ----------------------------------------------------------------- |
| Correctness | 80%    | 0.80 for exact match, 0.20 for close (scale error), 0.0 for wrong |
| Efficiency  | 10%    | Fewer turns = higher score. 1 turn = 0.10, 10 turns = 0.01        |
| Token Cost  | 10%    | Lower token usage = higher score                                  |

## Datasets

### Bank Marketing (750K rows)

- **Source**: UCI ML Repository (augmented)
- **Target**: Term deposit subscription (y = 0/1)
- **Features**: age, job, marital, education, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome

### Road Safety (500K rows)

- **Source**: UK Department for Transport (synthetic)
- **Target**: Accident risk (continuous 0-1)
- **Features**: road_type, num_lanes, curvature, speed_limit, lighting, weather, road_signs_present, time_of_day, num_reported_accidents

## Task Difficulty Levels

| Level | Complexity      | Example                                         |
| ----- | --------------- | ----------------------------------------------- |
| L4    | Quartile/binned | "Subscription rate in Q1 balance?"              |
| L5    | Multi-condition | "Rate for month='may' AND job='management'?"    |
| L6    | Nested extrema  | "In lowest subscription month, what's avg day?" |

## Build & Deploy

### Local Development

```bash
cd envs/codedark_env
pip install -e .

# Run server
python -m codedark_env.server.app
# Server at http://localhost:8000
```

### Docker

```bash
cd envs/codedark_env/server
docker build -t codedark:latest .
docker run -p 8000:8000 codedark:latest
```

### Deploy to HuggingFace

```bash
openenv push codedark_env
```

## API Endpoints

| Endpoint    | Method | Description            |
| ----------- | ------ | ---------------------- |
| `/`         | GET    | Landing page with docs |
| `/health`   | GET    | Health check           |
| `/reset`    | POST   | Reset for new episode  |
| `/step`     | POST   | Execute action         |
| `/state`    | GET    | Current state          |
| `/metadata` | GET    | Environment info       |
| `/schema`   | GET    | Type schemas           |
| `/docs`     | GET    | Interactive Swagger UI |

## Benchmark Results

Pre-benchmarked on 11+ models with 1,844 completions on the 25-task benchmark:

| Model            | Accuracy | Avg Turns | Cost/Task |
| ---------------- | -------- | --------- | --------- |
| Claude Opus 4.5  | 77.3%    | 4.2       | $0.89     |
| Qwen3 Max        | 46.7%    | 5.1       | $0.12     |
| Mistral Large    | 45.3%    | 5.8       | $0.18     |
| Llama 4 Maverick | 38.7%    | 6.2       | $0.08     |

Full leaderboard: https://www.analytics-rl.com

## Links

- **HuggingFace Space**: https://huggingface.co/spaces/albert-einstein-09/codedark
- **Live API**: https://albert-einstein-09-codedark.hf.space
- **Leaderboard**: https://www.analytics-rl.com
- **Original Benchmark**: https://github.com/vj-09/codeblue-env

## License

MIT License

## Author

**Vijay Athithya**

- GitHub: [@vj-09](https://github.com/vj-09)
- LinkedIn: [vijay-athithya](https://www.linkedin.com/in/vijay-athithya/)
