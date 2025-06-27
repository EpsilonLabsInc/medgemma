import json
import os
import gc
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from epsutils.dicom import dicom_utils
from epsutils.image import image_utils
from PIL import Image as PILImage

# ── CONFIG ────────────────────────────────────────────────────────────────────────────────
# INPUT_JSONL   = Path("/home/eric/projects/medgemma/data/first_1000_eval.jsonl")
# OUTPUT_JSONL  = Path("/home/eric/projects/medgemma/data/first_1000_eval_png.jsonl")
INPUT_JSONL   = Path("/home/eric/projects/medgemma/data/all_06082025_no_labels_eval.jsonl")
OUTPUT_JSONL  = Path("/home/eric/projects/medgemma/data/all_06082025_no_labels_eval_png.jsonl")

INPUT_JSONL   = Path("/home/eric/projects/medgemma/data/all_06082025_no_labels_test.jsonl")
OUTPUT_JSONL  = Path("/home/eric/projects/medgemma/data/all_06082025_no_labels_test_png.jsonl")

INPUT_JSONL   = Path("/home/eric/projects/medgemma/data/all_06082025_no_labels_train.jsonl")
OUTPUT_JSONL  = Path("/home/eric/projects/medgemma/data/all_06082025_no_labels_train_png.jsonl")

# PNG_ROOT      = Path("/home/eric/projects/medgemma/data/png")
PNG_ROOT      = Path("/mnt/training/png")

TARGET_SIZE   = (512, 512)
MAX_WORKERS   = os.cpu_count() or 4
print(f"Using {MAX_WORKERS} workers")
CHUNK_SIZE    = 10     # how many records per task batch (tune for your workload)
# ──────────────────────────────────────────────────────────────────────────────────────────

def process_record(line: str):
    rec = json.loads(line)
    new_paths = []

    # Make sure arr/pil always exist
    arr = None
    pil = None

    for dcm_path in rec["image"]:
        try:
            arr = dicom_utils.get_dicom_image_fail_safe(
                dcm_path,
                custom_windowing_parameters={"window_center": 0, "window_width": 0},
            )
            pil = image_utils.numpy_array_to_pil_image(arr, convert_to_rgb=True)
            pil = pil.resize(TARGET_SIZE, PILImage.Resampling.LANCZOS)

            rel = Path(dcm_path).relative_to("/mnt")
            out_png = (PNG_ROOT / rel).with_suffix(".png")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            pil.save(out_png)

        except Exception:
            # drop on any error
            return None

        finally:
            # close and release references safely
            if pil is not None:
                try:
                    pil.close()
                except Exception:
                    pass
            # instead of del, just reassign to None
            arr = None
            pil = None
            gc.collect()

        new_paths.append(str(out_png))

    rec["image"] = new_paths
    return rec


def main():
    # 1) Count lines for a proper tqdm total
    total = sum(1 for _ in INPUT_JSONL.open("r"))

    # 2) Open files & executor
    with INPUT_JSONL.open("r") as fin, OUTPUT_JSONL.open("w") as fout, \
         ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        # executor.map will pull lines from fin, dispatching in CHUNK_SIZE batches
        futures = executor.map(process_record, fin, chunksize=CHUNK_SIZE)

        # 3) Iterate & write results as they arrive, with a progress bar
        for processed in tqdm(futures, total=total, desc="Converting to PNG"):
            if processed is not None:
                fout.write(json.dumps(processed) + "\n")

if __name__ == "__main__":
    main()
