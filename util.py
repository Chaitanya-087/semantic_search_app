import os
import json
import glob

def build_text(record):
    parts = []
    for key in ["title", "main_category", "store"]:
        if record.get(key):
            parts.append(str(record[key]))

    for key in ["features", "description"]:
        if isinstance(record.get(key), list):
            parts.append(" ".join(record[key]))

    if isinstance(record.get("details"), dict):
        parts.append(" ".join(f"{k}: {v}" for k, v in record["details"].items()))

    return " ".join(parts)

def load_chunks(pattern):
    records = []
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    return records

def split_jsonl(input_path, parts=8, output_dir="chunks"):

    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r") as f:
        total = sum(1 for _ in f)

    chunk_size = total // parts
    print(f"Total: {total}, Chunk size: {chunk_size}")

    out_files = [
        open(os.path.join(output_dir, f"chunk_{i}.jsonl"), "w")
        for i in range(parts)
    ]

    current_idx = 0
    current_part = 0

    with open(input_path, "r") as f:
        for line in f:
            out_files[current_part].write(line)
            current_idx += 1

            if current_idx >= chunk_size and current_part < parts - 1:
                current_part += 1
                current_idx = 0

    for f in out_files:
        f.close()

    print("Splitting completed!")

if __name__ == '__main__':
    split_jsonl("data/products.jsonl", parts=32,output_dir="data/chunks")