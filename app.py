from flask import Flask, render_template, request
from search.engine import ProductSearchEngine

app = Flask(__name__)

engine = ProductSearchEngine(
    index_path="faiss.index",
    meta_path="metadata.json",
    chunk_pattern="data/chunk_test.jsonl"
)

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    if request.method == "POST":
        query = request.form.get("query", "")
        if query.strip():
            results = engine.search(query, top_k=10, gate=0.55)

    return render_template("index.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True, port=8501)
