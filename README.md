# In-Context Learning: Chain-of-Thought vs Tree-of-Thoughts

NLP project comparing CoT and ToT prompting strategies on the GSM8K math reasoning benchmark using Llama 3.1 8B via Groq API.

## Results

| Strategy | Accuracy | Samples |
|---|---|---|
| Zero-Shot CoT | **87.0%** | 100 |
| Few-Shot CoT (8-shot) | **85.0%** | 100 |
| Tree-of-Thoughts (BFS-2) | **13.3%** | 30 |

**Key finding:** ToT significantly underperforms CoT with a small (8B) model. The evaluator prompt requires strong meta-cognitive reasoning that 8B models cannot reliably perform, confirming that ToT is a large-model technique.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
python data/download_data.py
```

## Usage

```bash
# Zero-shot CoT (100 samples)
python src/run_experiments.py --strategy zero_shot_cot --n_samples 100

# Few-shot CoT (100 samples)
python src/run_experiments.py --strategy few_shot_cot --n_samples 100

# Tree-of-Thoughts (30 samples — takes ~90 min due to API rate limits)
python src/run_experiments.py --strategy tot --n_samples 30
```

## Project Structure

```
nlp-cot-tot/
├── data/
│   ├── download_data.py       # GSM8K download + answer extraction
│   └── few_shot_examples.json # 8-shot CoT examples (Wei et al. 2022)
├── prompts/                   # Prompt templates
├── src/
│   ├── llm_client.py          # Groq API wrapper with retry logic
│   ├── baseline_zero_shot_cot.py
│   ├── baseline_few_shot_cot.py
│   ├── tot_solver.py          # ToT BFS implementation
│   ├── evaluate.py            # Accuracy metric
│   └── run_experiments.py     # Unified CLI entry point
└── results/                   # Output JSON files
```

## Model

**Llama 3.1 8B** via [Groq API](https://console.groq.com). Get a free API key at console.groq.com.

## Dataset

GSM8K (Grade School Math 8K) — 8,792 math word problems requiring 2–8 reasoning steps.
Download: `python data/download_data.py`

## References

- Wei et al. (2022) — Chain-of-Thought Prompting Elicits Reasoning in LLMs
- Kojima et al. (2022) — Large Language Models are Zero-Shot Reasoners
- Yao et al. (2023) — Tree of Thoughts: Deliberate Problem Solving with LLMs
