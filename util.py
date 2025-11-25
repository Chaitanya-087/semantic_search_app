import os
import json
import glob
import math

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

    input_records = []

    # Read JSONL lines
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)     # <-- FIXED

            if not record.get("description"):
                continue

            input_records.append(record)

    total = len(input_records)
    chunk_size = math.ceil(total / parts)  # <-- FIXED (no zero division)

    print(f"Total: {total}, Chunk size: {chunk_size}")

    # Create output files
    out_files = [
        open(os.path.join(output_dir, f"chunk_{i}.jsonl"), "w", encoding="utf-8")
        for i in range(parts)
    ]

    current_part = 0
    current_count = 0

    for record in input_records:
        out_files[current_part].write(json.dumps(record) + "\n")  # <-- FIXED
        current_count += 1

        if current_count >= chunk_size and current_part < parts - 1:
            current_part += 1
            current_count = 0

    for f in out_files:
        f.close()

    print("Splitting completed!")
    
def minifiedJsonl(input_path, size=25000, output_file="products.json"):
    valid_records = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue 

            if not record.get("description"):
                continue

            title = record.get("title")
            description = " ".join(record.get("description")) if isinstance(record.get("description"), list) else record.get("description")
            price = 'N/A' if record.get("price") == None else record.get("price", "N/A")
            images = record.get("images", [])

            valid_records.append({
                "title": title,
                "description": description,
                "price": price,
                "images": images
            })

            if len(valid_records) >= size:
                break

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(valid_records, out, ensure_ascii=False, indent=2)

    print(f"Completed! Saved {len(valid_records)} records to {output_file}")


if __name__ == '__main__':
    # split_jsonl("data/products.jsonl", parts=32, output_dir="data/chunks")
    # minifiedJsonl("data/products.jsonl")
    pass
