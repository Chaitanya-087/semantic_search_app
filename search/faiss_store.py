import faiss
import numpy as np

class FAISSStore:
    @staticmethod
    def build_and_save(embeddings, path):
        # Normalize embeddings to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)  # avoid division by zero

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # use inner product
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, path)
        print("Saved FAISS:", path)

    @staticmethod
    def load(path):
        return FAISSIndex(faiss.read_index(path))


class FAISSIndex:
    def __init__(self, index):
        self.index = index

    def search(self, query_emb, top_k):
        # Normalize query embedding
        query_emb = np.array(query_emb, dtype="float32")
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        query_emb = np.expand_dims(query_emb, axis=0)  # shape (1, dim)

        distances, indices = self.index.search(query_emb, top_k)
        # distances here are actually inner product scores (higher = more similar)
        return {
            "scores": distances[0].tolist(),  # renamed to "scores" for clarity
            "indices": indices[0].tolist()
        }
