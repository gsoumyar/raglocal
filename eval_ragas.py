"""
RAGAS Evaluation Harness
========================
Runs the golden test set against the RAG pipeline and scores each answer
on three dimensions using local Llama as judge:

  Faithfulness  - is every claim grounded in the retrieved context?
  Relevancy     - does the answer address the question asked?
  Correctness   - does the answer match the known ground truth?

Usage:
  1. Start the server: uvicorn main:app
  2. Upload tools.pdf (or your target document)
  3. Run: python3 eval_ragas.py
  4. Compare scores across configurations (CR on/off, chunk sizes, etc.)
"""

import json
import os
import sys
import requests
from datetime import datetime

# ── Config ──
API_URL = "http://127.0.0.1:8000"
OLLAMA_URL = "http://localhost:11434/api/chat"
GOLDEN_TEST_PATH = "golden_test.json"
RESULTS_PATH = "eval_results.json"


def ask_pipeline(question: str) -> dict:
    """Send a question to our RAG pipeline and get answer + retrieved context."""
    try:
        resp = requests.post(f"{API_URL}/ask", json={"text": question}, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to the server. Is uvicorn running?")
        sys.exit(1)
    except requests.exceptions.Timeout:
        return {"answer": "TIMEOUT", "retrieved_context": [], "sources": []}


def llm_judge(prompt: str) -> float:
    """Ask Llama to return a single float score. Returns 0.0 on any failure."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }, timeout=60)
        raw = resp.json()["message"]["content"].strip()
        # Extract the first float-like token from the response
        for token in raw.split():
            token = token.strip(".,;:!?")
            try:
                score = float(token)
                return min(max(score, 0.0), 1.0)
            except ValueError:
                continue
        return 0.0
    except Exception:
        return 0.0


def judge_faithfulness(question: str, answer: str, context: str) -> float:
    """Is every claim in the answer supported by the retrieved context?"""
    prompt = f"""You are an evaluation judge. Given a question, an answer, and the source context
that was used to generate the answer, determine what fraction of the claims
in the answer are actually supported by the source context.

Question: {question}

Answer: {answer}

Source Context:
{context}

Instructions:
1. Identify each distinct factual claim made in the answer.
2. For each claim, check if it is directly supported by the source context.
3. Calculate: (number of supported claims) / (total number of claims).
4. Return ONLY a single number between 0.0 and 1.0. Nothing else."""
    return llm_judge(prompt)


def judge_relevancy(question: str, answer: str) -> float:
    """Does the answer actually address the question that was asked?"""
    prompt = f"""You are an evaluation judge. Given a question and an answer,
determine how relevant the answer is to the question.

Question: {question}

Answer: {answer}

A score of 1.0 means the answer perfectly addresses the question.
A score of 0.0 means the answer is completely irrelevant.
Return ONLY a single number between 0.0 and 1.0. Nothing else."""
    return llm_judge(prompt)


def judge_correctness(answer: str, ground_truth: str) -> float:
    """Does the answer capture the key facts from the known ground truth?"""
    prompt = f"""You are an evaluation judge. Compare the generated answer to the ground truth.

Generated Answer: {answer}

Ground Truth: {ground_truth}

How much of the ground truth information is captured in the generated answer?
A score of 1.0 means all key facts from the ground truth are present.
A score of 0.0 means nothing from the ground truth appears.
Return ONLY a single number between 0.0 and 1.0. Nothing else."""
    return llm_judge(prompt)


def run_evaluation():
    """Run the full evaluation suite."""
    # Load golden test set
    with open(GOLDEN_TEST_PATH) as f:
        test_set = json.load(f)

    print("=" * 60)
    print("RAGAS EVALUATION HARNESS")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Questions: {len(test_set)}")
    print("=" * 60)

    results = []
    all_faithfulness = []
    all_relevancy = []
    all_correctness = []

    for i, test in enumerate(test_set):
        q = test["question"]
        gt = test["ground_truth"]

        print(f"\n--- Question {i+1}/{len(test_set)}: {q}")

        # Get answer from pipeline
        pipeline_result = ask_pipeline(q)
        answer = pipeline_result.get("answer", "")
        retrieved_context = pipeline_result.get("retrieved_context", [])
        sources = pipeline_result.get("sources", [])

        # Build context string for faithfulness check
        context_str = "\n\n".join(retrieved_context) if retrieved_context else answer

        print(f"    Answer: {answer[:120]}...")

        # Judge each dimension
        faith = judge_faithfulness(q, answer, context_str)
        relev = judge_relevancy(q, answer)
        correct = judge_correctness(answer, gt)

        all_faithfulness.append(faith)
        all_relevancy.append(relev)
        all_correctness.append(correct)

        print(f"    Faithfulness: {faith:.2f} | Relevancy: {relev:.2f} | Correctness: {correct:.2f}")

        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "sources": sources,
            "faithfulness": faith,
            "relevancy": relev,
            "correctness": correct
        })

    # Summary
    avg_f = sum(all_faithfulness) / len(all_faithfulness) if all_faithfulness else 0
    avg_r = sum(all_relevancy) / len(all_relevancy) if all_relevancy else 0
    avg_c = sum(all_correctness) / len(all_correctness) if all_correctness else 0

    print("\n" + "=" * 60)
    print("OVERALL SCORES")
    print("=" * 60)
    print(f"  Faithfulness:  {avg_f:.2f}")
    print(f"  Relevancy:     {avg_r:.2f}")
    print(f"  Correctness:   {avg_c:.2f}")
    print(f"  Questions:     {len(test_set)}")
    print("=" * 60)

    # Save detailed results for comparison across runs
    eval_record = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "faithfulness": round(avg_f, 3),
            "relevancy": round(avg_r, 3),
            "correctness": round(avg_c, 3),
            "num_questions": len(test_set)
        },
        "details": results
    }

    # Append to results file (keeps history of all runs for comparison)
    history = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            history = json.load(f)
    history.append(eval_record)
    with open(RESULTS_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDetailed results saved to {RESULTS_PATH}")
    print(f"Run count: {len(history)} (compare across configurations)")

    return eval_record


if __name__ == "__main__":
    run_evaluation()