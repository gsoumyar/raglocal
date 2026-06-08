import requests

response = requests.post("http://localhost:11434/api/chat", json={
    "model": "llama3.2",
    "messages": [
        {"role": "user", "content": "What is Newton's first law? Answer in two sentences."}
    ],
    "stream": False
})

data = response.json()
print(data["message"]["content"])