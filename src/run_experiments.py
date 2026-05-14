import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.download_data import extract_answer
import baseline_zero_shot_cot
import baseline_few_shot_cot
from tot_solver import solve
from evaluate import accuracy, print_results


def load_test(path: str = "data/gsm8k_test.json", n: int = 100, seed: int = 42) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    random.seed(seed)
    return random.sample(data, min(n, len(data)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["zero_shot_cot", "few_shot_cot", "tot"], required=True)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    args = parser.parse_args()

    test_data = load_test(n=args.n_samples)

    if args.strategy == "zero_shot_cot":
        results = baseline_zero_shot_cot.run(test_data, model=args.model)
    elif args.strategy == "few_shot_cot":
        results = baseline_few_shot_cot.run(test_data, model=args.model)
    elif args.strategy == "tot":
        results = []
        for item in test_data:
            out = solve(item["question"], model=args.model)
            gold = extract_answer(item["answer"])
            out["gold"] = gold
            out["correct"] = out["predicted"] == gold
            results.append(out)

    print_results(args.strategy, results)
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.strategy}_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
