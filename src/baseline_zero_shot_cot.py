import re
import json
from llm_client import chat

PROMPT_TEMPLATE = "Q: {question}\nA: Let's think step by step."


def extract_final_answer(text: str) -> float | None:
    numbers = re.findall(r"[\d,]+\.?\d*", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def run(questions: list[dict], model: str = "llama-3.1-8b-instant") -> list[dict]:
    results = []
    for item in questions:
        prompt = PROMPT_TEMPLATE.format(question=item["question"])
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
