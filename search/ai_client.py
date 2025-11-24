import numpy as np
import requests

class GeminiClient:
    def __init__(self, api_key=""):
        self.api_key = api_key
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        self.headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def embed(self, text):
        payload = {"model": "models/gemini-embedding-001",
                   "content": {"parts": [{"text": text}]}}
        r = requests.post(self.url, headers=self.headers, json=payload)
        r.raise_for_status()
        return np.array(r.json()["embedding"]["values"], dtype="float32")

    def embed_batch(self, texts):
        embs = []
        for t in texts:
            embs.append(self.embed(t))
        return np.array(embs, dtype="float32")

class OpenAIClient:
    def __init__(self, api_key):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def embed(self, text):
        r = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(r.data[0].embedding, dtype="float32")

    def embed_batch(self, texts):
        r = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return np.array([d.embedding for d in r.data], dtype="float32")
