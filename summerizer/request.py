import requests

texts = ["This is a test sentence.", "Another sentence."]
response = requests.post("http://localhost:8001/embed", json={"texts": texts})
embeddings = response.json()["embeddings"]

print(len(embeddings), len(embeddings[0]),embeddings)  # 2 x 1536
