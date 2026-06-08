from FlagEmbedding import BGEM3FlagModel

print("Loading BGE-M3...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("Model loaded.")

test_texts = [
    "Pozidriv screwdriver has additional ribs between the arms",
    "Phillips screwdriver has a cross-shaped tip",
    "A centrifuge spins samples at high speed"
]

# BGE-M3 produces both dense and sparse in one call
output = model.encode(test_texts, return_dense=True, return_sparse=True)

print(f"\nDense shape: {output['dense_vecs'].shape}")
print(f"Sparse keys (first text): {list(output['lexical_weights'][0].keys())[:10]}...")

# Test similarity
from numpy import dot
from numpy.linalg import norm

q = model.encode(["what is a Pozidriv?"], return_dense=True)['dense_vecs'][0]
for i, text in enumerate(test_texts):
    d = output['dense_vecs'][i]
    sim = dot(q, d) / (norm(q) * norm(d))
    print(f"  '{text[:50]}...' → similarity: {sim:.4f}")

print("\nBGE-M3 working.")