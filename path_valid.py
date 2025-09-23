import json
import os
from tqdm import tqdm

jsonl_file = "/home/eric/projects/medgemma/data/all_09222025_test.jsonl"
missing_files = []

missing_files = []

# Count total lines (dicts)
with open(jsonl_file, "r") as f:
    total_lines = sum(1 for _ in f)

print("start checking files...")
with open(jsonl_file, "r") as f:
    for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Checking files", unit="dict")):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"Invalid JSON at line {line_num+1}")
            continue

        for path in record.get("image", []):
            if not os.path.exists(path):
                print(path)
                missing_files.append((line_num+1, path))

print("\n=== Missing Files Report ===")
if missing_files:
    for line, path in missing_files:
        print(f"Line {line}: {path}")
    print(f"\nTotal missing files: {len(missing_files)}")
else:
    print("✅ All files exist")