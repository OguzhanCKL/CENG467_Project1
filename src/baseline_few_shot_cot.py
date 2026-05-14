import re
import json
from llm_client import chat


def load_few_shot_examples(path: str = "data/few_shot_examples.json") -> str:
    with open(path) as f:
        examples = json.load(f)
    formatted = ""
    for ex in examples:
        formatted += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    return formatted.strip()


def extract_final_answer(text: str) -> float | None:
    numbers = re.findall(r"[\d,]+\.?\d*", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def run(questions: list[dict], model: str = "llama-3.1-8b-instant") -> list[dict]:
    few_shot = load_few_shot_examples()
    results = []
    for item in questions:
        prompt = f"{few_shot}\n\nQ: {item['question']}\nA:"
        response = chat(prompt, model=model)
        predicted = extract_final_answer(response)
        gold = extract_final_answer(item["answer"])
        results.append({
            "question": item["question"],
            "gold": gold,
            "predicted": predicted,
            "correct": predicted == gold,
            "response": response,
        })
    return results
