from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

print("Loading reranker...")
rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
rerank_model.eval()
print("Reranker loaded.")

question = "What is a Pozidriv screwdriver?"

candidates = [
    "Flathead Screwdriver has a single flat blade that fits into slotted screw heads.",
    "Phillips Screwdriver has a cross-shaped tip that fits Phillips head screws.",
    "Pozidriv Screwdriver is similar to Phillips but with additional ribs between the arms. Common in European-made furniture.",
    "A centrifuge spins samples at high speed to separate components by density.",
    "Torx Screwdriver has a six-pointed star-shaped tip found in electronics and automotive parts."
]

# Reranker takes (question, candidate) pairs
pairs = [[question, c] for c in candidates]

with torch.no_grad():
    inputs = rerank_tokenizer(
        pairs, padding=True, truncation=True,
        return_tensors="pt", max_length=512
    )
    scores = rerank_model(**inputs).logits.view(-1).float().tolist()

print(f"\nQuestion: {question}\n")
for score, text in sorted(zip(scores, candidates), reverse=True):
    print(f"  Score {score:>8.4f} — {text[:70]}...")