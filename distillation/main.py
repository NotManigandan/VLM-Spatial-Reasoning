import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import os
import math
import argparse
from dotenv import load_dotenv
import wandb
from dataclasses import asdict
from config import DistillationConfig
from utils.load_model import load_models_and_processor
from utils.spacethinker_data import prepare_datasets, prepare_dataloaders
from loss.logit_distillation import logit_distillation_loss
from loss.attn_distillation import attn_distillation_loss


def parse_args():
    default_cfg = DistillationConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_id", default=default_cfg.teacher_model_id)
    parser.add_argument("--teacher_adapter_path", default=default_cfg.teacher_adapter_path)
    parser.add_argument("--student_model_id", default=default_cfg.student_model_id)
    parser.add_argument("--dataset_id", default=default_cfg.dataset_id)
    parser.add_argument("--lora_r", type=int, default=default_cfg.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=default_cfg.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=default_cfg.lora_dropout)
    parser.add_argument("--target_modules", default=",".join(default_cfg.target_modules), help="Comma-separated list of target modules for LoRA")
    parser.add_argument("--num_train_epochs", type=int, default=default_cfg.num_train_epochs)
    parser.add_argument("--train_batch_size", type=int, default=default_cfg.train_batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=default_cfg.eval_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=default_cfg.gradient_accumulation_steps)
    parser.add_argument("--learning_rate", type=float, default=default_cfg.learning_rate)
    parser.add_argument("--warmup_ratio", type=float, default=default_cfg.warmup_ratio)
    parser.add_argument("--max_grad_norm", type=float, default=default_cfg.max_grad_norm)
    parser.add_argument("--have_scheduler", type=bool, default=default_cfg.have_scheduler)
    parser.add_argument("--temperature", type=float, default=default_cfg.temperature)
    parser.add_argument("--loss_mode", type=str, default=default_cfg.loss_mode)
    parser.add_argument("--layer_map_mode", type=str, default=default_cfg.layer_map_mode)
    parser.add_argument("--ce_weightage", type=float, default=default_cfg.ce_weightage)
    parser.add_argument("--attn_weightage", type=float, default=default_cfg.attn_weightage)
    parser.add_argument("--output_dir", default=default_cfg.output_dir)
    parser.add_argument("--eval_steps", type=int, default=default_cfg.eval_steps)
    parser.add_argument("--log_steps", type=int, default=default_cfg.log_steps)
    args = parser.parse_args()
    return DistillationConfig(
        teacher_model_id=args.teacher_model_id,
        teacher_adapter_path=args.teacher_adapter_path,
        student_model_id=args.student_model_id,
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
        max_grad_norm=args.max_grad_norm,
        have_scheduler=args.have_scheduler,
        temperature=args.temperature,
        loss_mode=args.loss_mode,
        layer_map_mode=args.layer_map_mode,
        ce_weightage=args.ce_weightage,
        attn_weightage=args.attn_weightage,
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps
    )

def evaluate(teacher_model, student_model, eval_loader, cfg, device):
    student_model.eval()
    final_loss = []
    with torch.no_grad():
        for minibatch in tqdm(eval_loader, desc="Validation"):
            minibatch = {i: j.to(device) for i, j in minibatch.items()}
            labels = minibatch.pop("labels")
            teacher_outputs = teacher_model(**minibatch, output_attentions=True, return_dict=True)
            student_outputs = student_model(**minibatch, output_attentions=True, return_dict=True)
            if cfg.loss_mode == "attention":
                loss = attn_distillation_loss(student_logits = student_outputs.logits, teacher_logits = teacher_outputs.logits, student_attentions = student_outputs.attentions, teacher_attentions = teacher_outputs.attentions, labels = labels, attn_weightage=cfg.attn_weightage, ce_weightage = cfg.ce_weightage, include_logit_loss = True, student_loss=None, l_mapping = None, mode = cfg.layer_map_mode, temperature = cfg.temperature)
            else:
                loss = logit_distillation_loss(student_outputs.logits, teacher_outputs.logits, labels, temperature=cfg.temperature, ce_weightage=cfg.ce_weightage)
            final_loss.append(loss.item())
    student_model.train()
    return sum(final_loss)/len(final_loss)


def train_model():
    # https://huggingface.co/learn/llm-course/en/chapter3/4
    # https://huggingface.co/learn/llm-course/chapter7/6
    cfg = parse_args()
    print(f"Distillation Configuration:\n{asdict(cfg)}")
    load_dotenv()
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    run = wandb.init(
            project="spacethinker-lora",
            name="Qwen2.5-VL-3B-Attn-Distillation",
            config=asdict(cfg)
        )
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_output_dir = os.path.join(cfg.output_dir, "Best")
    final_output_dir = os.path.join(cfg.output_dir, "Final")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    step_track = 0
    best_eval_loss = float("inf")
    train_dataset, eval_dataset = prepare_datasets(cfg)
    teacher_model, student_model, processor = load_models_and_processor(cfg)
    train_loader, eval_loader = prepare_dataloaders(cfg, processor, train_dataset, eval_dataset, use_collate_fn = True)

    steps_in_epoch = math.ceil(len(train_loader)/cfg.gradient_accumulation_steps)
    training_steps = math.ceil(steps_in_epoch*cfg.num_train_epochs)
    warmup_steps = math.ceil(cfg.warmup_ratio*training_steps)

    optimizer = torch.optim.AdamW([p for p in student_model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    if cfg.have_scheduler and warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.001, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=training_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    elif cfg.have_scheduler and warmup_steps <= 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=training_steps)  
    else:
        scheduler = None

    student_model.train()
    for epoch in range(cfg.num_train_epochs):
        print(f"[Epoch {epoch + 1}]")
        running_loss = 0.0
        for step, minibatch in enumerate(tqdm(train_loader, desc="Train")):
            minibatch = {i: j.to(device) for i, j in minibatch.items()}
            labels = minibatch.pop("labels")
            # labels = minibatch["labels"]
            with torch.no_grad():
                teacher_outputs = teacher_model(**minibatch, output_attentions=True, return_dict=True)
            student_outputs = student_model(**minibatch, output_attentions=True, return_dict=True)
            # old_loss = logit_distillation_loss(student_outputs.logits, teacher_outputs.logits, labels, temperature=cfg.temperature, ce_weightage=cfg.ce_weightage, student_loss=student_outputs.loss)
            if cfg.loss_mode == "attention":
                old_loss = attn_distillation_loss(student_logits = student_outputs.logits, teacher_logits = teacher_outputs.logits, student_attentions = student_outputs.attentions, teacher_attentions = teacher_outputs.attentions, labels =labels, attn_weightage=cfg.attn_weightage, ce_weightage = cfg.ce_weightage, include_logit_loss = True, student_loss=None, l_mapping = None, mode = cfg.layer_map_mode, temperature = cfg.temperature)
            else:
                old_loss = logit_distillation_loss(student_outputs.logits, teacher_outputs.logits, labels, temperature=cfg.temperature, ce_weightage=cfg.ce_weightage, student_loss=None)
            loss = old_loss/cfg.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in student_model.parameters() if p.requires_grad], cfg.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                step_track += 1
            
                if step_track % cfg.eval_steps == 0:
                    eval_loss = evaluate(teacher_model, student_model, eval_loader, cfg, device)
                    print(f"Validation Loss at step {step_track} is {eval_loss}")
                    wandb.log({"eval/loss": eval_loss, "eval/global_step": step_track})
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        print(f"Best Model with val loss of {best_eval_loss}. Saving it at {best_output_dir}")
                        student_model.save_pretrained(best_output_dir)
                        processor.save_pretrained(best_output_dir)
                if step_track > 0 and step_track % cfg.log_steps == 0:
                    wandb.log(
                        {
                            "train/loss": old_loss.item(),
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/global_step": step_track,
                        }
                    )
                
            running_loss += loss.item() * cfg.gradient_accumulation_steps

        # to handle case where total steps is not multiple of gradient_accumulation_steps
        if (step+1) % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_([p for p in student_model.parameters() if p.requires_grad], cfg.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            step_track += 1
        print(f"Epoch {epoch+1} Train Loss: {running_loss / len(train_loader)}")
        eval_loss = evaluate(teacher_model, student_model, eval_loader, cfg, device)
        print(f"Validation Loss at step {step_track} is {eval_loss}")
        wandb.log({"eval/loss": eval_loss, "eval/global_step": step_track})
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            print(f"Best Model with val loss of {best_eval_loss}. Saving it at {best_output_dir}")
            student_model.save_pretrained(best_output_dir)
            processor.save_pretrained(best_output_dir)

    student_model.save_pretrained(final_output_dir)
    processor.save_pretrained(final_output_dir)
    wandb.finish()

if __name__ == "__main__":
    train_model()