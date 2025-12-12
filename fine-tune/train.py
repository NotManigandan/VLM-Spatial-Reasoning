import re
import argparse
from dataclasses import dataclass, field
from typing import List

import torch
import wandb
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer


def extract_question(raw_text: str) -> str:
    pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>"
    m = re.search(pattern, raw_text, re.DOTALL)
    return m.group(1).strip() if m else raw_text.strip()

def format_data_spacethinker(sample):
    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are VL-Thinking U+1F914, a helpful assistant with excellent reasoning ability.\n"
                    "A user asks you a question, and you should try to solve it."
                    "You should first think about the reasoning process in the mind and then provides the user with the answer.\n"
                    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
                )
            }
        ]
    }
    formatted = [system_message]

    user_msg = {"role": "user", "content": []}
    question = extract_question(sample.get("input", ""))
    if question:
        user_msg["content"].append({"type": "text", "text": question})
    images = sample.get("images") or []
    if images:
        user_msg["content"].append({"type": "image", "image": images[0]})
    formatted.append(user_msg)

    if sample.get("output"):
        formatted.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["output"]}]
        })
    return formatted


def collate_fn(examples, processor):
    # examples: list of formatted samples (list of message dicts)
    texts = [processor.apply_chat_template(sample, tokenize=False) for sample in examples]
    image_batches = [process_vision_info(sample)[0] for sample in examples]
    batch = processor(text=texts, images=image_batches, return_tensors="pt", padding=True)
    batch = {k: v.cpu() for k, v in batch.items()}

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_token_ids = (
        [151652, 151653, 151655]
        if hasattr(processor, "image_processor")
        else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    )
    for tid in image_token_ids:
        labels[labels == tid] = -100

    batch["labels"] = labels
    return batch


@dataclass
class TrainingConfig:
    model_id: str = "UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B"
    dataset_id: str = "remyxai/SpaceThinker"
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    num_train_epochs: int = 3
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    output_dir: str = "spacethinker-lora"
    wandb_project: str = "spacethinker-lora"
    wandb_run_name: str = "spacethinker_run"


def parse_args() -> TrainingConfig:
    default_cfg = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train a VL Spacethinker model with LoRA")
    parser.add_argument("--model_id", default=default_cfg.model_id)
    parser.add_argument("--dataset_id", default=default_cfg.dataset_id)
    parser.add_argument("--lora_r", type=int, default=default_cfg.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=default_cfg.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=default_cfg.lora_dropout)
    parser.add_argument(
        "--target_modules",
        default=','.join(default_cfg.target_modules),
        help="Comma-separated list of target modules for LoRA"
    )
    parser.add_argument("--num_train_epochs", type=int, default=default_cfg.num_train_epochs)
    parser.add_argument("--train_batch_size", type=int, default=default_cfg.train_batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=default_cfg.eval_batch_size)
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=default_cfg.gradient_accumulation_steps
    )
    parser.add_argument("--learning_rate", type=float, default=default_cfg.learning_rate)
    parser.add_argument("--warmup_ratio", type=float, default=default_cfg.warmup_ratio)
    parser.add_argument("--output_dir", default=default_cfg.output_dir)
    parser.add_argument("--wandb_project", default=default_cfg.wandb_project)
    parser.add_argument("--wandb_run_name", default=default_cfg.wandb_run_name)

    args = parser.parse_args()
    return TrainingConfig(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


def prepare_datasets(cfg: TrainingConfig):
    print(f"Loading dataset: {cfg.dataset_id}…")
    raw_train = load_dataset(cfg.dataset_id, split="train")
    raw_eval = load_dataset(cfg.dataset_id, split="test")

    print("Formatting train samples…")
    train_ds = [format_data_spacethinker(s) for s in tqdm(raw_train, desc="Train")]
    print("Formatting eval samples…")
    eval_ds = [format_data_spacethinker(s) for s in tqdm(raw_eval, desc="Eval")]

    return train_ds, eval_ds


def prepare_model_and_optimizer(cfg: TrainingConfig):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb
    )
    processor = AutoProcessor.from_pretrained(cfg.model_id)

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_model = get_peft_model(model, peft_cfg).to(device)
    peft_model.print_trainable_parameters()
    return peft_model, processor, peft_cfg


def main():
    cfg = parse_args()
    train_ds, eval_ds = prepare_datasets(cfg)
    model, processor, peft_cfg = prepare_model_and_optimizer(cfg)

    sft_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="constant",
        logging_steps=10,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=cfg.warmup_ratio,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=True,
        report_to="wandb",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    sft_args.remove_unused_columns = False

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=sft_args,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=lambda ex: collate_fn(ex, processor),
        peft_config=peft_cfg,
        processing_class=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()