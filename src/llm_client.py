import os
import time
from openai import OpenAI, RateLimitError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

DEFAULT_MODEL = "llama-3.1-8b-instant"


def chat(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    for attempt in range(8):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            time.sleep(8)
            return response.choices[0].message.content.strip()
        except RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"  Rate limit hit, waiting {wait}s...")
            time.sleep(wait)
        except APIConnectionError:
            wait = 10 * (attempt + 1)
            print(f"  Connection error, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Failed after 8 attempts.")
