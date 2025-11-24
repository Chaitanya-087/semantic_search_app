import faiss
import numpy as np

class FAISSStore:
    @staticmethod
    def build_and_save(embeddings, path):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, path)
        print("Saved FAISS:", path)

    @staticmethod
    def load(path):
        return FAISSIndex(faiss.read_index(path))

class FAISSIndex:
    def __init__(self, index):
        self.index = index

    def search(self, query_emb, top_k):
        query_emb = np.array([query_emb], dtype="float32")
        distances, indices = self.index.search(query_emb, top_k)
        return {
            "distances": distances[0].tolist(),
            "indices": indices[0].tolist()
        }
