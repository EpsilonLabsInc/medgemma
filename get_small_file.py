import json
from tqdm import tqdm

INPUT_PATH  = "/home/eric/projects/medgemma/data/all_06082025_no_labels_eval.jsonl"
OUTPUT_PATH = "/home/eric/projects/medgemma/data/first_1000_eval.jsonl"

INPUT_PATH = "/home/eric/projects/medgemma/data/all_06082025_no_labels_test_png.jsonl"
OUTPUT_PATH = "/home/eric/projects/medgemma/data/first_1000_test_png.jsonl"


MAX_LINES   = 1000

with open(INPUT_PATH, "r") as fin, open(OUTPUT_PATH, "w") as fout:
    for i, line in enumerate(tqdm(fin, total=MAX_LINES, desc="Copying lines")):
        if i >= MAX_LINES:
            break

        record = json.loads(line)
        # ── If you want the entire dict:
        fout.write(json.dumps(record) + "\n")

        # ── If you only want to keep a text snippet (e.g. last assistant reply),
        #    uncomment these two lines and comment out the two lines above:
        #
        # text = record["conversations"][-1]["value"]
        # fout.write(json.dumps({"text": text}) + "\n")
