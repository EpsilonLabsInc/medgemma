import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "0"

import torch

torch._dynamo.reset()
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.verbose = False
torch._dynamo.config.cache_size_limit = 0

from transformers import AutoModel, AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from datasets import load_dataset, Image
from epsutils.dicom import dicom_utils
from epsutils.image import image_utils
from typing import Any, List, Dict
from peft import LoraConfig
from PIL import Image
from PIL import Image as PILImage

import wandb
from datetime import datetime


model_id = "google/medgemma-4b-it"

# Check if GPU supports bfloat16
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError(
        "GPU does not support bfloat16, please use a GPU that supports bfloat16."
    )

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
)

# model_kwargs["quantization_config"] = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
#     bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
# )

model_med_gemma = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)


path = "/home/eric/projects/InternVL-3x/internvl_chat/pretrained/InternVL3-8B/"
intern3_8b = (
    AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    )
)


model_med_gemma.model.vision_tower = intern3_8b.vision_model

from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector

config = Gemma3Config.from_json_file("../medgemma-4b-it/config.json")
print(config.vision_config.hidden_size)
config.vision_config.hidden_size = 1024
print(config.vision_config.hidden_size)


proj_new = Gemma3MultiModalProjector(config)
proj_new = proj_new.to(dtype=torch.bfloat16, device=model_med_gemma.device)

model_med_gemma.model.multi_modal_projector = proj_new
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

# Use right padding to avoid issues during training
processor.tokenizer.padding_side = "right"


data_files = {
    # "validation": "/home/eric/projects/medgemma/data/first_1000_eval.jsonl",
    # "validation": "/home/eric/projects/medgemma/data/first_1000_eval_png.jsonl",
    # "validation": "/home/eric/projects/medgemma/data/all_06082025_no_labels_eval_png.jsonl",
    # "validation": "/home/eric/projects/medgemma/data/all_06082025_no_labels_eval.jsonl",
    # "train": "/home/eric/projects/medgemma/data/all_06082025_no_labels_train.jsonl"

    "test": "/home/eric/projects/medgemma/data/all_09222025_test_png.jsonl",
}

ds = load_dataset(
    "json",
    data_files=data_files,
)  # :contentReference[oaicite:0]{index=0}


def make_messages(example):
    # 1) Strip out the "<image>" tokens from the human prompt:
    human_val = example["conversations"][0]["value"]
    prompt = human_val.replace("<image>", "").strip()

    # 2) Build the 'user' content array: one {"type":"image"} per DICOM + the text block
    user_content = []
    for _ in example["image"]:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": prompt})

    # 3) Build the 'assistant' content array from the GPT reply
    gpt_val = example["conversations"][-1]["value"]
    assistant_content = [{"type": "text", "text": gpt_val}]

    # 4) Assign the new messages field
    example["messages"] = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return example


# # Apply it to your whole dataset:
ds = ds.map(make_messages)

print(f"Dataset size: {len(ds['test'])} records")

ds = ds.map(lambda ex: {"n_images": len(ex["image"])})
ds = ds.filter(lambda ex: ex["n_images"] <= 5)
ds = ds.remove_columns("n_images")

print(f"Filtered dataset size: {len(ds['test'])} records")


def collate_fn(examples: List[Dict[str, Any]]):
    texts: List[str] = []
    images: List[List] = []

    for example in examples:
        pil_imgs = []
        for png_path in example["image"]:
            pil = PILImage.open(png_path).convert("RGB")
            pil_imgs.append(pil)

        images.append(pil_imgs)

        txt = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        ).strip()
        texts.append(txt)

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()
    boi = processor.tokenizer.special_tokens_map["boi_token"]
    boi_id = processor.tokenizer.convert_tokens_to_ids(boi)

    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == boi_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)


if __name__ == "__main__":
    from trl import SFTConfig
    from trl import SFTTrainer

    rank = int(os.environ.get("RANK", 0))
    is_main = rank == 0

    # Disable W&B in all non‐main processes
    if not is_main:
        # you can also use "disabled" instead of "offline"
        os.environ["WANDB_MODE"] = "offline"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if is_main:
        wandb.login()
        project_name = "chimera_medgemma-intern-0922"
        run_name = f"run-{timestamp}"

        wandb.init(project=project_name, name=run_name)

    args = SFTConfig(
        output_dir=f"training/test-{timestamp}",  # Directory and Hub repository id to save the model to
        num_train_epochs=5,  # Number of training epochs
        per_device_train_batch_size=1,  # Batch size per device during training
        # per_device_eval_batch_size=2,  # Batch size per device during evaluation
        gradient_accumulation_steps=16,  # Number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # Enable gradient checkpointing to reduce memory usage
        optim="adamw_torch_fused",  # Use fused AdamW optimizer for better performance
        logging_steps=2,  # Number of steps between logs
        save_strategy="steps",  # Save checkpoint every epoch
        save_steps=500,  # Number of steps between saves
        # eval_strategy="epoch",  # Evaluate every `eval_steps` or `step`
        # eval_steps=50,  # Number of steps between evaluations
        learning_rate=1e-5,  # Learning rate based on QLoRA paper
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=0.3,  # Max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # Warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",  # Use linear learning rate scheduler
        push_to_hub=True,  # Push model to Hub
        report_to="wandb",  # Report metrics to tensorboard
        # run_name="medgemma-4b-it-123",  # Name of the run in wandb
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # Set gradient checkpointing to non-reentrant to avoid issues
        dataset_kwargs={
            "skip_prepare_dataset": True
        },  # Skip default dataset preparation to preprocess manually
        remove_unused_columns=False,  # Columns are unused for training but needed for data collator
        label_names=["labels"],  # Input keys that correspond to the labels
        deepspeed="ds_config.json",
        dataloader_num_workers=8,
        dataloader_pin_memory=False,  # Pin memory for faster data loading
        torch_compile=False,  # Enable PyTorch compilation for performance
    )

    trainer = SFTTrainer(
        model=model_med_gemma,
        args=args,
        train_dataset=ds["test"],
        # eval_dataset=ds["validation"],
        # eval_dataset=data["validation"].shuffle().select(range(200)),  # Use subset of validation set for faster run
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    print(model_med_gemma)

    trainer.train()
