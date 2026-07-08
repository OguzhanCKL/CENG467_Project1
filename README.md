# In-Context Learning: Chain-of-Thought vs Tree-of-Thoughts

CENG467 project comparing in-context learning strategies — Zero-Shot CoT, Few-Shot CoT, Self-Consistency, and Tree-of-Thoughts — on the GSM8K math reasoning benchmark. Primary model: **Llama 3.1 8B** (Groq API). Cross-model replication: **Claude Haiku 4.5** (Anthropic API).

## Reports

- [`CENG467_Final_Report.pdf`](CENG467_Final_Report.pdf) — final report (Springer LNCS format, 15 pages)
- [`Chain_of_Thought_vs_Tree_of_Thoughts.pdf`](Chain_of_Thought_vs_Tree_of_Thoughts.pdf) — IEEE conference format (6 pages)
- [`CENG467_Final_Report_IEEE.tex`](CENG467_Final_Report_IEEE.tex) — LaTeX source of the IEEE version

## Results

All strategies evaluated on the same 100 GSM8K test problems (seed 42). Accuracies are gold-corrected: one GSM8K gold-label error (carnival problem, 2280 → 2180) is fixed, so file-level raw accuracies can be 1 pp lower.

| Strategy | Llama 3.1 8B | Claude Haiku 4.5 |
|---|---|---|
| Zero-Shot CoT (v2, format-hinted) | **95.0%** | 97.0% |
| Few-Shot CoT (8-shot) | 85.0% | 98.0% |
| Self-Consistency (5 paths, majority vote) | 92.0% | **99.0%** |
| Tree-of-Thoughts (depth=3, branch=2) | 81.0% | 98.0% |
| Tree-of-Thoughts (depth=6, branch=2) | 85.0% | 98.0% |

**Key findings**

- **Strategy gap collapses with model capability:** strategies span 14 pp on the 8B model but only 2 pp on Haiku 4.5. Plain Zero-Shot CoT on Haiku (97%) beats the best 8B strategy (SC, 92%).
- **ToT depth matters only for the weaker model:** going from depth 3 to depth 6 gains +4.0 pp on Llama at ~2.1× cost, but +0.0 pp on Haiku (ceiling effect). Cohen's κ between the two depths: 0.011 on Llama (near-random disagreement) vs 1.000 on Haiku (identical predictions).
- **Self-Consistency's verifier gap:** on Llama, Pass@5 is 99% while majority vote reaches 92% — the correct answer is almost always sampled but not always selected. On Haiku the gap disappears.
- **Hard problems dominate the error budget:** on Haiku, four of five strategies are perfect on Easy+Medium; every remaining error is in the Hard bucket. On Llama, Zero-Shot CoT leads the Hard bucket.
- **ToT is not a free win for small models:** at temperature 0 with a small backbone, tree search mostly re-ranks similar reasoning paths; careful prompt/output formatting (ZS-CoT v2) is far more cost-effective.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY (and ANTHROPIC_API_KEY for the Haiku notebooks) to .env
python data/download_data.py
```

## Usage

```bash
# Zero-shot / few-shot CoT (100 samples)
python src/run_experiments.py --strategy zero_shot_cot --n_samples 100
python src/run_experiments.py --strategy few_shot_cot --n_samples 100

# Self-Consistency (5 sampled paths per problem, temperature 0.7)
python src/self_consistency.py --n_samples 100 --n_runs 5

# Tree-of-Thoughts (depth=6, checkpointed; run in 3 parts for Groq free-tier quota)
python src/run_tot_fixed.py --part 1
python src/run_tot_fixed.py --part 2
python src/run_tot_fixed.py --part 3
```

Claude Haiku 4.5 replication runs live in `notebooks/` (self-contained, Anthropic SDK; the CoT/SC strategies use the Batch API for 50% cost reduction).

## Project Structure

```
├── CENG467_Final_Report.pdf              # Final report (LNCS)
├── Chain_of_Thought_vs_Tree_of_Thoughts.pdf  # IEEE-format report
├── CENG467_Final_Report_IEEE.tex         # IEEE LaTeX source
├── data/
│   ├── download_data.py        # GSM8K download + answer extraction
│   └── few_shot_examples.json  # 8-shot CoT examples (Wei et al. 2022)
├── prompts/                    # Prompt templates
├── src/
│   ├── llm_client.py           # Groq API wrapper with retry logic
│   ├── baseline_zero_shot_cot.py
│   ├── baseline_few_shot_cot.py
│   ├── self_consistency.py     # SC with majority vote
│   ├── run_tot_fixed.py        # ToT BFS (depth=6, checkpointed)
│   ├── evaluate.py             # Accuracy metric
│   ├── make_figs.py            # Report figures
│   └── run_experiments.py      # CLI entry point for CoT baselines
├── notebooks/                  # Haiku 4.5 cross-model runs + ablations
└── results/                    # Output JSON files (Llama + Haiku)
```

## Models

- **Llama 3.1 8B** (`llama-3.1-8b-instant`) via [Groq API](https://console.groq.com) — free tier is sufficient.
- **Claude Haiku 4.5** (`claude-haiku-4-5`) via [Anthropic API](https://console.anthropic.com) — full 5-strategy replication cost $3.26.

## Dataset

GSM8K (Grade School Math 8K) — 8,792 math word problems requiring 2–8 reasoning steps.
Download: `python data/download_data.py`

## Authors

Sercan Utku Temelci · Oğuzhan Çakal — İzmir Institute of Technology (İYTE), CENG467

## References

- Wei et al. (2022) — Chain-of-Thought Prompting Elicits Reasoning in LLMs
- Kojima et al. (2022) — Large Language Models are Zero-Shot Reasoners
- Wang et al. (2022) — Self-Consistency Improves Chain of Thought Reasoning
- Yao et al. (2023) — Tree of Thoughts: Deliberate Problem Solving with LLMs
- Cobbe et al. (2021) — Training Verifiers to Solve Math Word Problems (GSM8K)
