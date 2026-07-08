"""
Fixed Tree-of-Thoughts runner.

Fixes applied vs the original ToT implementation:
  1. depth=6 instead of 3 (most GSM8K problems need 4-8 steps)
  2. Final extraction call after BFS (model reads full reasoning, outputs #### N)
  3. extract_answer prioritises #### X pattern (consistent with CoT scripts)

Checkpoint system: saves after every problem -> safe to Ctrl+C and resume.
3-part split: run one part per session to stay within Groq free-tier quota.
  Part 1: problems  1-34
  Part 2: problems 35-67
  Part 3: problems 68-100

Usage:
  python src/run_tot_fixed.py --part 1
  python src/run_tot_fixed.py --part 2
  python src/run_tot_fixed.py --part 3
"""

import json
import random
import re
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from llm_client import chat

BASE      = os.path.join(os.path.dirname(__file__), "..")
DATA      = os.path.join(BASE, "data")
RESULTS   = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

MODEL      = "llama-3.1-8b-instant"
CHECKPOINT = os.path.join(RESULTS, "tot_fixed_results.json")

PART_SLICES = {1: (0,  34),
               2: (34, 67),
               3: (67, 100)}

# ------------------------------------------------------------------ prompts

THOUGHT_PROMPT = """Solve this math problem step by step. Generate the next single reasoning step only.

Problem: {problem}
Steps so far: {steps}

Next step:"""

EVALUATE_PROMPT = """Problem: {problem}
Reasoning so far: {steps}
Candidate next step: {candidate}

Is this step leading toward the correct solution? Reply with one word: sure, maybe, or impossible."""

# FIX 2: dedicated extraction prompt — model sees full reasoning, returns #### N
EXTRACT_PROMPT = """Problem: {problem}
Reasoning:
{reasoning}

Based on the reasoning above, what is the final numeric answer?
Write only the number on the last line in this exact format:
#### <number>"""


# ------------------------------------------------------------------ helpers

def extract_answer(text: str):
    """FIX 3: prioritise #### X pattern, fall back to last number."""
    m = re.search(r"####\s*([\d,\.]+)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    nums = re.findall(r"[\d,]+\.?\d*", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def gold_answer(text: str):
    m = re.search(r"####\s*([\d,\.]+)", text)
    return float(m.group(1).replace(",", "")) if m else None


# ------------------------------------------------------------------ solver

def solve(problem: str, branching: int = 2, depth: int = 6) -> dict:
    """FIX 1: depth=6. FIX 2: final extraction call."""
    beam = [""]

    for _ in range(depth):
        candidates = []
        for path in beam:
            for _ in range(branching):
                thought = chat(
                    THOUGHT_PROMPT.format(problem=problem, steps=path),
                    model=MODEL, temperature=0.7
                )
                score = chat(
                    EVALUATE_PROMPT.format(problem=problem, steps=path, candidate=thought),
                    model=MODEL, temperature=0.0
                )
                candidates.append((path + "\n" + thought, score))

        order = {"sure": 0, "maybe": 1, "impossible": 2}
        candidates.sort(key=lambda x: order.get(x[1].strip().lower().split()[0], 3))
        beam = [c[0] for c in candidates[:branching]]

    best_path = beam[0]

    # FIX 2: ask model to read full reasoning and output #### answer
    final_resp = chat(
        EXTRACT_PROMPT.format(problem=problem, reasoning=best_path),
        model=MODEL, temperature=0.0
    )
    predicted = extract_answer(final_resp)
    if predicted is None:          # fallback: last number in reasoning
        predicted = extract_answer(best_path)

    return {"reasoning": best_path, "final_response": final_resp, "predicted": predicted}


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, choices=[1, 2, 3], required=True,
                        help="Part to run: 1 (problems 1-34), 2 (35-67), 3 (68-100)")
    parser.add_argument("--branching", type=int, default=2)
    parser.add_argument("--depth",     type=int, default=6)
    args = parser.parse_args()

    start, end = PART_SLICES[args.part]

    # Always sample the same 100 problems (seed=42)
    with open(os.path.join(DATA, "gsm8k_test.json"), encoding="utf-8") as f:
        all_data = json.load(f)
    random.seed(42)
    all_100 = random.sample(all_data, 100)
    part_samples = all_100[start:end]

    # Load checkpoint
    results        = []
    done_questions = set()
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as f:
            ckpt = json.load(f)
        results        = ckpt.get("results", [])
        done_questions = {r["question"] for r in results}
        print(f"  Checkpoint: {len(results)} problems already done.")

    print(f"  Part {args.part}: problems {start+1}–{end}  "
          f"(depth={args.depth}, branching={args.branching})")

    for i, item in enumerate(part_samples, start + 1):
        if item["question"] in done_questions:
            print(f"  [{i}/100] skipping (already done)...", end="\r", flush=True)
            continue

        print(f"  [{i}/100] solving...", end="\r", flush=True)

        out  = solve(item["question"], branching=args.branching, depth=args.depth)
        gold = gold_answer(item["answer"])

        results.append({
            "question":       item["question"],
            "gold":           gold,
            "predicted":      out["predicted"],
            "correct":        out["predicted"] == gold,
            "reasoning":      out["reasoning"],
            "final_response": out["final_response"],
        })

        correct_so_far = sum(r["correct"] for r in results)
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump({
                "strategy":  f"tot_fixed_depth{args.depth}_branch{args.branching}",
                "model":     MODEL,
                "n_done":    len(results),
                "accuracy":  round(correct_so_far / len(results) * 100, 1),
                "results":   results,
            }, f, indent=2, ensure_ascii=False)

    correct  = sum(r["correct"] for r in results)
    n        = len(results)
    print(f"\n  Part {args.part} complete.  "
          f"Running total: {correct}/{n}  Accuracy={correct/n*100:.1f}%")


if __name__ == "__main__":
    main()
