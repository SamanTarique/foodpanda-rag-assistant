"""
10 test questions chalata hai, Gemini se khud RAG-answer generate karwata hai,
phir ek DOOSRI Gemini call se us answer ko evaluate (Correct/Partial/Incorrect
+ Hallucination Yes/No) karwata hai, aur evaluation.csv mein save karta hai.

Free tier rate limit (~10 requests/minute) se bachne ke liye har question
ke baad delay hai, aur agar phir bhi 429 (rate limit) aaye to crash nahi
hota - thodi dair wait karke retry karta hai.

Run: project root se -> python -m tests.test_pipeline
"""
import re
import csv
import sys
import time
from pathlib import Path

# IMPORTANT: sys.path insert rag.pipeline import se PEHLE hona chahiye,
# warna 'python tests/test_pipeline.py' (bina -m ke) chalane par
# "ModuleNotFoundError: No module named 'rag'" aayega.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from google import genai
from google.genai import errors
from config import GEMINI_API_KEY, GEMINI_MODEL
from rag.pipeline import generate_answer

EVAL_CSV = ROOT_DIR / "logs" / "evaluation.csv"
client = genai.Client(api_key=GEMINI_API_KEY)

DELAY_BETWEEN_QUESTIONS = 15   # seconds - free tier ~10 RPM se bachne ke liye
RATE_LIMIT_WAIT = 65           # 429 aane par itni dair wait karo (RPM window reset)
MAX_RETRIES = 3


def call_with_retry(fn, *args, **kwargs):
    """Gemini call karta hai; 429 aaye to wait karke retry karta hai (crash nahi hota)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except errors.ClientError as e:
            if e.code == 429 and attempt < MAX_RETRIES:
                print(f"\n⚠️ Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                      f"Waiting {RATE_LIMIT_WAIT}s before retry...")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            raise
    return None


def evaluate(question, expected, actual):
    prompt = f"""Compare the chatbot answer with the expected answer.

Question:
{question}

Expected Answer:
{expected}

Chatbot Answer:
{actual}

Return ONLY these two lines, nothing else:
Score: Correct / Partially Correct / Incorrect
Hallucination: Yes / No
"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()


def parse_evaluation(result: str) -> tuple[str, str]:
    """Gemini ke evaluation response se Score aur Hallucination alag nikalta hai."""
    score_match = re.search(r"Score:\s*(.+)", result)
    halluc_match = re.search(r"Hallucination:\s*(Yes|No)", result)
    score = score_match.group(1).strip() if score_match else "Unknown"
    hallucination = halluc_match.group(1).strip() if halluc_match else "Unknown"
    return score, hallucination


def run_tests():
    with open(EVAL_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    correct = 0
    hallucinations = 0
    tested = 0

    for i, row in enumerate(rows):
        question = row["question"]
        expected = row["expected_answer_summary"]

        print("\n" + "=" * 60)
        print(f"Question {i + 1}/{len(rows)}: {question}")

        try:
            actual = call_with_retry(generate_answer, question)
        except errors.ClientError as e:
            print(f"\nGemini API Error (giving up after retries): {e}")
            break

        print("\nAnswer:", actual)

        try:
            eval_result = call_with_retry(evaluate, question, expected, actual)
        except errors.ClientError as e:
            print(f"\nGemini API Error (giving up after retries): {e}")
            break

        score, hallucination = parse_evaluation(eval_result)
        print("Evaluation:", score, "| Hallucination:", hallucination)

        row["chatbot_answer"] = actual
        row["accuracy"] = score
        row["hallucination"] = hallucination

        tested += 1
        if score == "Correct":
            correct += 1
        if hallucination == "Yes":
            hallucinations += 1

        # Save progress after EVERY question (agar beech mein rukna pade to data safe rahe)
        with open(EVAL_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        if i < len(rows) - 1:  # last question ke baad wait ki zaroorat nahi
            print(f"(waiting {DELAY_BETWEEN_QUESTIONS}s before next question...)")
            time.sleep(DELAY_BETWEEN_QUESTIONS)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Tested             : {tested}/{len(rows)}")
    if tested > 0:
        print(f"Correct            : {correct}  ({correct/tested*100:.2f}%)")
        print(f"Hallucinations     : {hallucinations}  ({hallucinations/tested*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()