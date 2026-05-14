import re
from llm_client import chat

THOUGHT_PROMPT = """Solve this math problem step by step. Generate the next single reasoning step only.

Problem: {problem}
Steps so far: {steps}

Next step:"""

EVALUATE_PROMPT = """Problem: {problem}
Reasoning so far: {steps}
Candidate next step: {candidate}

Is this step leading toward the correct solution? Reply with one word: sure, maybe, or impossible."""


def generate_thoughts(problem: str, steps: str, n: int = 3, model: str = "llama-3.1-8b-instant") -> list[str]:
    thoughts = []
    for _ in range(n):
        prompt = THOUGHT_PROMPT.format(problem=problem, steps=steps)
        thoughts.append(chat(prompt, model=model, temperature=0.7))
    return thoughts


def evaluate_thought(problem: str, steps: str, candidate: str, model: str = "llama-3.1-8b-instant") -> str:
    prompt = EVALUATE_PROMPT.format(problem=problem, steps=steps, candidate=candidate)
    return chat(prompt, model=model, temperature=0.0)


def extract_final_answer(text: str) -> float | None:
    numbers = re.findall(r"[\d,]+\.?\d*", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def solve(problem: str, branching: int = 2, depth: int = 3, model: str = "llama-3.1-8b-instant") -> dict:
    beam = [""]  # current paths
    for _ in range(depth):
        candidates = []
        for path in beam:
            thoughts = generate_thoughts(problem, path, n=branching, model=model)
            for t in thoughts:
                score = evaluate_thought(problem, path, t, model=model)
                candidates.append((path + "\n" + t, score))
        order = {"sure": 0, "maybe": 1, "impossible": 2}
        candidates.sort(key=lambda x: order.get(x[1].strip().lower().split()[0], 3))
        beam = [c[0] for c in candidates[:branching]]
    best_path = beam[0]
    return {
        "problem": problem,
        "reasoning": best_path,
        "predicted": extract_final_answer(best_path),
    }
