import numpy as np
import requests
import base64


class GeminiClient:
    def __init__(self, api_key=""):
        self.api_key = api_key
        self.headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def embed(self, text):
        embed_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        payload = {"model": "models/gemini-embedding-001",
                   "content": {"parts": [{"text": text}]}}
        r = requests.post(embed_url, headers=self.headers, json=payload)
        r.raise_for_status()
        return np.array(r.json()["embedding"]["values"], dtype="float32")

    def embed_batch(self, texts):
        embs = []
        for t in texts:
            embs.append(self.embed(t))
        return np.array(embs, dtype="float32")

    def describe_image(self, img_bytes: bytes):
            flash_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            payload = {
                "model": "models/gemini-2.5-flash",
                "contents": [
                    {
                        "image": {
                            "mime_type": "image/jpeg",  # or: "image/png"
                            "data": img_b64
                        },
                        "parts": [{"text": "Describe the content of this image"}]
                    }
                ],
                "generationConfig": {
                    # you can set temperature, maxOutputTokens, etc.
                    "temperature": 0.0,
                    "maxOutputTokens": 128
                }
            }

            r = requests.post(flash_url, headers=self.headers, json=payload)
            r.raise_for_status()
            resp = r.json()
            print(resp)

            description = []
            for candidate in resp.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if part.get("text"):
                        description.append(part["text"])
            return " ".join(description)



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



if __name__ == "__main__":
    import os
    import requests

    client = GeminiClient(os.getenv("GEMINI_API_KEY"))

    url = "https://m.media-amazon.com/images/I/411S0HZd5WL._AC_SR38,50_.jpg"
    response = requests.get(url)
    response.raise_for_status()
    img_bytes = response.content

    try:
        description = client.describe_image(img_bytes)
        print("Description:", description)
    except NotImplementedError:
        print("Image description not supported on Tier 1 Gemini accounts.")
