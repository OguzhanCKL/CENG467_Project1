from datasets import load_dataset
import json, re, os


def download_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main")
    train = list(dataset["train"])
    test = list(dataset["test"])
    os.makedirs("data", exist_ok=True)
    with open("data/gsm8k_train.json", "w") as f:
        json.dump(train, f, indent=2)
    with open("data/gsm8k_test.json", "w") as f:
        json.dump(test, f, indent=2)
    print(f"Train: {len(train)}, Test: {len(test)}")


def extract_answer(solution: str) -> float | None:
    match = re.search(r"####\s*([\d,\.]+)", solution)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


if __name__ == "__main__":
    download_gsm8k()
