import os, json, sys
from .ai_client import GeminiClient
from .faiss_store import FAISSStore
from util import build_text, load_chunks
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import numpy as np

load_dotenv(override=True)
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

class ProductSearchEngine:
    def __init__(self,chunk_pattern,
                 index_path="faiss.index",
                 meta_path="texts.json"):
        
        self.index_path = index_path
        self.meta_path = meta_path
        self.chunk_pattern = chunk_pattern

        self.client = GeminiClient(api_key=GEMINI_API_KEY) 

        # Load or build index
        if os.path.exists(index_path) and os.path.exists(meta_path):
            print("Loading existing FAISS index...")
            self.texts = json.load(open(meta_path))
            self.index = FAISSStore.load(index_path)
        else:
            print("Building embeddings + FAISS index...")
            self.texts = load_chunks(self.chunk_pattern)
            texts_for_embed = [build_text(t) for t in self.texts]
            total = len(texts_for_embed)
            done = 0
            embeddings = [None] * total
            failed_indices = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(self.client.embed, text): idx for idx, text in enumerate(texts_for_embed)}

                for future in as_completed(futures):
                    idx = futures[future]
                    done += 1
                    try:
                        emb = future.result()
                        embeddings[idx] = emb
                    except Exception as e:
                        print(f"Embedding failed for text: {futures[future]} → {e}")
                        failed_indices.append(idx)
                        embeddings[idx] = None
                    sys.stdout.write(
                        f"\rCompleted {done}/{total} embeddings ({(done/total)*100:.2f}%)"
                    )
                    sys.stdout.flush()
            print()
            successful_indices = [i for i, e in enumerate(embeddings) if e is not None]
            if not successful_indices:
                raise RuntimeError("No embeddings succeeded. Check error logs.")

            emb_matrix = np.array([embeddings[i] for i in successful_indices], dtype="float32")
            aligned_texts = [self.texts[i] for i in successful_indices]

            json.dump(aligned_texts, open(meta_path, "w"), indent=2)

            FAISSStore.build_and_save(emb_matrix, index_path)

            self.texts = aligned_texts
            self.index = FAISSStore.load(index_path)

    def search(self, query, gate, top_k=5):
        qemb = self.client.embed(query)
        hits = self.index.search(qemb, top_k)
        
        indices = hits["indices"]
        scores = hits["scores"] 

        docs_with_scores = [
            (self.texts[i], score)
            for i, score in zip(indices, scores)
            if score >= gate
        ]
        
        return docs_with_scores
