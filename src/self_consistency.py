"""
Self-Consistency + Few-Shot CoT baseline.

Runs few-shot CoT N_RUNS times per problem with temperature=0.7,
then selects the answer by majority vote (Wang et al., 2022).

Usage:
    python src/self_consistency.py --n_samples 100 --n_runs 5
"""

import json
import random
import re
import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from llm_client import chat

BASE    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

MODEL       = "llama-3.1-8b-instant"
TEMPERATURE = 0.7


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_answer(text: str):
    m = re.search(r"####\s*([\d,\.]+)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    nums = re.findall(r"[\d,]+\.?\d*", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def format_examples(examples):
    return "\n\n".join(f"Q: {e['question']}\nA: {e['answer']}" for e in examples)


def majority_vote(answers: list):
    """Return the most common numeric answer; None if all are None."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def run(n_samples=100, n_runs=5, seed=42):
    test_data = load_json(os.path.join(DATA, "gsm8k_test.json"))
    random.seed(seed)
    test_data = random.sample(test_data, min(n_samples, len(test_data)))

    canonical    = load_json(os.path.join(DATA, "few_shot_examples.json"))
    few_shot_str = format_examples(canonical)

    path = os.path.join(RESULTS, "self_consistency_results.json")

    # Resume from checkpoint if exists
    results = []
    done_questions = set()
    if os.path.exists(path):
        checkpoint = load_json(path)
        results = checkpoint.get("results", [])
        done_questions = {r["question"] for r in results}
        print(f"  Resuming from checkpoint: {len(results)} problems already done.")

    total = len(test_data)

    for i, item in enumerate(test_data, 1):
        if item["question"] in done_questions:
            print(f"  [{i}/{total}] skipping (already done)...", end="\r", flush=True)
            continue

        prompt = f"{few_shot_str}\n\nQ: {item['question']}\nA:"
        print(f"  [{i}/{total}] sampling {n_runs} paths...", end="\r", flush=True)

        sampled_answers = []
        sampled_responses = []
        for run_idx in range(n_runs):
            response = chat(prompt, model=MODEL, temperature=TEMPERATURE)
            ans      = extract_answer(response)
            sampled_answers.append(ans)
            sampled_responses.append(response)

        predicted = majority_vote(sampled_answers)
        gold      = extract_answer(item["answer"])

        results.append({
            "question":         item["question"],
            "gold":             gold,
            "predicted":        predicted,
            "correct":          predicted == gold,
            "sampled_answers":  sampled_answers,
            "vote_counts":      dict(Counter(
                                    a for a in sampled_answers if a is not None)),
            "responses":        sampled_responses,
        })

        # save after every problem so we never lose progress
        correct_so_far = sum(r["correct"] for r in results)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "strategy":  f"self_consistency_{n_runs}runs",
                "model":     MODEL,
                "n_samples": i,
                "n_runs":    n_runs,
                "accuracy":  round(correct_so_far / i * 100, 1),
                "results":   results,
            }, f, indent=2, ensure_ascii=False)

    correct  = sum(r["correct"] for r in results)
    accuracy = correct / total * 100
    print(f"  Self-Consistency ({n_runs} runs): {correct}/{total}  "
          f"Accuracy={accuracy:.1f}%     ")
    return results, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_runs",    type=int, default=5)
    args = parser.parse_args()

    results, accuracy = run(n_samples=args.n_samples, n_runs=args.n_runs)

    out = {
        "strategy":  f"self_consistency_{args.n_runs}runs",
        "model":     MODEL,
        "n_samples": args.n_samples,
        "n_runs":    args.n_runs,
        "accuracy":  round(accuracy, 1),
        "results":   results,
    }
    path = os.path.join(RESULTS, "self_consistency_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved -> {path}")


if __name__ == "__main__":
    main()
