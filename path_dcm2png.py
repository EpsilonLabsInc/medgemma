import json
import os

from tqdm import tqdm


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))  # Parse each line as a JSON object
    print(f"{file_path} has {len(data)} records")
    return data


paths_map = {
    "gradient_01JUL2025": "/mnt/all-data/sfs-gradient-new/01JUL2025/",
    "gradient_09JAN2025": "/mnt/all-data/sfs-gradient-chest/09JAN2025/",
    "gradient_13JAN2025": "/mnt/all-data/sfs-gradient-nochest/13JAN2025/deid/",
    "gradient_16AUG2024": "/mnt/all-data/sfs-gradient-nochest/16AUG2024/",
    "gradient_20DEC2024": "/mnt/all-data/sfs-gradient-chest/20DEC2024/",
    "gradient_22JUL2024": "/mnt/all-data/sfs-gradient-chest/22JUL2024/",
    "simonmed": "/mnt/all-data/sfs-simonmed/",
    "segmed_batch_1": "/mnt/all-data/sfs-segmed-1/",
    "segmed_batch_2": "/mnt/all-data/sfs-segmed-2/",
    "segmed_batch_3": "/mnt/all-data/sfs-segmed-34/segmed_3/",
    "segmed_batch_4": "/mnt/all-data/sfs-segmed-34/segmed_4/",
    "segmed_batch_5": "/mnt/all-data/sfs-segmed567/segmed_5/",
    "segmed_batch_6": "/mnt/all-data/sfs-segmed567/segmed_6/",
    "segmed_batch_7": "/mnt/all-data/sfs-segmed567/segmed_7/",
    "segmed_batch_8": "/mnt/all-data/sfs-segmed89/batch8/",
    "segmed_batch_9": "/mnt/all-data/sfs-segmed89/batch9/",
    "segmed_batch_10": "/mnt/all-data/sfs-segmed10/",
    "segmed_batch_11": "/mnt/all-data/sfs-segmed11/",
    "segmed_batch_12": "/mnt/all-data/sfs-segmed12/",
    "segmed_batch_13": "/mnt/all-data/sfs-segmed13/",
    "segmed_batch_14": "/mnt/all-data/sfs-segmed14/",
    "segmed_batch_15": "/mnt/all-data/sfs-segmed15/",
    "segmed_batch_16": "/mnt/all-data/sfs-segmed16/",
}

png_paths_map = {
    "gradient_01JUL2025": "/mnt/all-data/png/512x512/gradient-new/01JUL2025/",
    "gradient_09JAN2025": "/mnt/all-data/png/512x512/gradient/09JAN2025/",
    "gradient_13JAN2025": "/mnt/all-data/png/512x512/gradient/13JAN2025/deid/",
    "gradient_16AUG2024": "/mnt/all-data/png/512x512/gradient/16AUG2024/",
    "gradient_20DEC2024": "/mnt/all-data/png/512x512/gradient/20DEC2024/",
    "gradient_22JUL2024": "/mnt/all-data/png/512x512/gradient/22JUL2024/",
    "simonmed": "/mnt/all-data/png/512x512/simonmed/",
    "segmed_batch_1": "/mnt/all-data/png/512x512/segmed/batch1/",
    "segmed_batch_2": "/mnt/all-data/png/512x512/segmed/batch2/",
    "segmed_batch_3": "/mnt/all-data/png/512x512/segmed/batch3/",
    "segmed_batch_4": "/mnt/all-data/png/512x512/segmed/batch4/",
    "segmed_batch_5": "/mnt/all-data/png/512x512/segmed/batch5/",
    "segmed_batch_6": "/mnt/all-data/png/512x512/segmed/batch6/",
    "segmed_batch_7": "/mnt/all-data/png/512x512/segmed/batch7/",
    "segmed_batch_8": "/mnt/all-data/png/512x512/segmed/batch8/",
    "segmed_batch_9": "/mnt/all-data/png/512x512/segmed/batch9/",
    "segmed_batch_10": "/mnt/all-data/png/512x512/segmed/batch10/",
    "segmed_batch_11": "/mnt/all-data/png/512x512/segmed/batch11/",
    "segmed_batch_12": "/mnt/all-data/png/512x512/segmed/batch12/",
    "segmed_batch_13": "/mnt/all-data/png/512x512/segmed/batch13/",
    "segmed_batch_14": "/mnt/all-data/png/512x512/segmed/batch14/",
    "segmed_batch_15": "/mnt/all-data/png/512x512/segmed/batch15/",
    "segmed_batch_16": "/mnt/all-data/png/512x512/segmed/batch16/",
}




def convert_dcm_to_png(data, paths_map, png_paths_map):
    for item in tqdm(data, desc="Converting paths", unit="row"):
        source = item["source"]
        if source not in paths_map:
            continue  # skip if source not mapped

        dcm_prefix = paths_map[source]
        png_prefix = png_paths_map[source]

        new_images = []
        for img_path in item["image"]:
            # replace prefix
            if img_path.startswith(dcm_prefix):
                img_path = img_path.replace(dcm_prefix, png_prefix, 1)

            # replace suffix
            if img_path.endswith(".dcm"):
                img_path = img_path[:-4] + ".png"

            # check if png exists
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"PNG file does not exist: {img_path}")

            new_images.append(img_path)

        item["image"] = new_images
    return data


def save_jsonl(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for row in tqdm(data, desc="Saving JSONL", unit="row"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    jsonl_test = "/home/eric/projects/medgemma/data/all_09222025_test.jsonl"

    data_test = load_jsonl(jsonl_test)

    data_test = convert_dcm_to_png(data_test, paths_map, png_paths_map)

    print("✅ All PNG files verified and paths updated.")
    print(data_test[0]["image"])

    output_jsonl = "/home/eric/projects/medgemma/data/all_09222025_test_png.jsonl"
    save_jsonl(data_test, output_jsonl)