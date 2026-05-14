def accuracy(results: list[dict]) -> float:
    correct = sum(1 for r in results if r.get("correct"))
    return correct / len(results) if results else 0.0


def print_results(strategy: str, results: list[dict]):
    acc = accuracy(results)
    print(f"{strategy}: {acc:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")
