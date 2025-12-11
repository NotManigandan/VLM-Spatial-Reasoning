from dataclasses import dataclass, field
from typing import List

@dataclass
class DistillationConfig:
    teacher_model_id: str = "UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B"
    teacher_adapter_path: str = "/home/ubuntu/10623_dev_workspace/Qwen2.5-VL-7B"
    student_model_id: str = "UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B"
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
    max_grad_norm: float = 0.3
    have_scheduler: bool = True
    temperature: float = 1.5
    loss_mode: str = "attention"
    layer_map_mode: str = "spread"
    ce_weightage: float = 0.5
    attn_weightage: float = 0.5 
    output_dir: str = "/home/ubuntu/10623_dev_workspace/Qwen2.5VL-3B-Attn-Distilled"
    eval_steps: int = 500
    log_steps: int = 10