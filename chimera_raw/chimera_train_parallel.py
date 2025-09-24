import os

# Set environment variables before importing torch
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "0"

import torch
import torch.distributed as dist

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


def setup_distributed():
    """Initialize distributed training environment"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun sets these automatically
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        # Initialize the process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        return rank, world_size, local_rank
    else:
        # Single GPU training
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


model_id = "google/medgemma-4b-it"

# Setup distributed training
rank, world_size, local_rank = setup_distributed()
is_main = rank == 0

# Only print on main process to avoid cluttered output
if is_main:
    print(f"Distributed training setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")

# Check if GPU supports bfloat16
if torch.cuda.get_device_capability(local_rank)[0] < 8:
    raise ValueError(
        f"GPU {local_rank} does not support bfloat16, please use a GPU that supports bfloat16."
    )

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    # device_map=f"cuda:{local_rank}",  # Explicitly set device for this process
)

# Load model on the correct device
model_med_gemma = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
model_med_gemma = model_med_gemma.to(f"cuda:{local_rank}")
# from torch.nn.parallel import DistributedDataParallel as DDP
# model_med_gemma = DDP(model_med_gemma, device_ids=[local_rank])

path = "/home/eric/projects/InternVL-3x/internvl_chat/pretrained/InternVL3-8B/"
intern3_8b = (
    AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        # device_map=f"cuda:{local_rank}",
    )
)
intern3_8b = intern3_8b.to(f"cuda:{local_rank}")
# intern3_8b = DDP(intern3_8b, device_ids=[local_rank])

model_med_gemma.model.vision_tower = intern3_8b.vision_model

# model_med_gemma.module.model.vision_tower = intern3_8b.module.vision_model

from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector

config = Gemma3Config.from_json_file("../medgemma-4b-it/config.json")
if is_main:
    print(config.vision_config.hidden_size)
config.vision_config.hidden_size = 1024
if is_main:
    print(config.vision_config.hidden_size)

proj_new = Gemma3MultiModalProjector(config)
proj_new = proj_new.to(dtype=torch.bfloat16, device=f"cuda:{local_rank}")

model_med_gemma.model.multi_modal_projector = proj_new
# model_med_gemma.module.model.multi_modal_projector = proj_new
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

# Use right padding to avoid issues during training
processor.tokenizer.padding_side = "right"

data_files = {
    "test": "/home/eric/projects/medgemma/data/all_09222025_test_png.jsonl",
}

ds = load_dataset(
    "json",
    data_files=data_files,
)

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

# Apply it to your whole dataset:
ds = ds.map(make_messages)

if is_main:
    print(f"Dataset size: {len(ds['test'])} records")

ds = ds.map(lambda ex: {"n_images": len(ex["image"])})
ds = ds.filter(lambda ex: ex["n_images"] <= 5)
ds = ds.remove_columns("n_images")

if is_main:
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

    # Don't move to device in collate_fn - let the trainer handle device placement
    # Moving tensors to device in multiprocessing workers causes CUDA re-initialization errors

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

    # Disable W&B in all non-main processes
    if not is_main:
        os.environ["WANDB_MODE"] = "offline"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Initialize wandb only on main process
    if is_main:
        wandb.login()
        project_name = "chimera_medgemma-intern-0924"
        run_name = f"run-{timestamp}"
        wandb.init(project=project_name, name=run_name)

    args = SFTConfig(
        output_dir=f"training/test-{timestamp}",
        num_train_epochs=5,
        per_device_train_batch_size=4,  # This is per GPU now
        gradient_accumulation_steps=16,  # May want to adjust this for distributed training
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=2,
        save_strategy="epoch",
        learning_rate=1e-5,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        push_to_hub=True if is_main else False,  # Only push from main process
        report_to="wandb" if is_main else None,  # Only report from main process
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
        dataset_kwargs={
            "skip_prepare_dataset": True
        },
        remove_unused_columns=False,
        label_names=["labels"],
        deepspeed="ds_config.json",
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues with CUDA
        dataloader_pin_memory=False,
        torch_compile=False,
        # Distributed training specific settings
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,  # Important for distributed training
        save_only_model=True,  # Save only model weights to avoid issues
    )

    trainer = SFTTrainer(
        model=model_med_gemma,
        args=args,
        train_dataset=ds["test"],
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    if is_main:
        print(model_med_gemma)
        print(args)
        print(f"Starting distributed training on {world_size} GPUs")

    try:
        trainer.train()
    finally:
        # Clean up distributed training
        cleanup_distributed()
        
    if is_main:
        print("Training completed successfully!")
