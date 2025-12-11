import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


def load_models_and_processor(cfg):
    teacher_processor = AutoProcessor.from_pretrained(cfg.teacher_model_id)
    teacher_bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    teacher_base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.teacher_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=teacher_bnb,
        attn_implementation="eager"
    )
    teacher_model = PeftModel.from_pretrained(teacher_base_model, cfg.teacher_adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_adapter_path, use_fast=False)
    teacher_processor.tokenizer = tokenizer
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    student_bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    student_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.student_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=student_bnb,
        attn_implementation="eager"
    )
    teacher_vocab_size = teacher_model.config.vocab_size
    student_base.resize_token_embeddings(teacher_vocab_size) # https://github.com/QwenLM/Qwen3/issues/29, https://github.com/QwenLM/Qwen3/issues/147, https://github.com/QwenLM/Qwen3/issues/466
    student_base.config.vocab_size = teacher_vocab_size
    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )
    student_model = get_peft_model(student_base, peft_cfg)
    student_model.print_trainable_parameters()

    # https://medium.com/@manindersingh120996/practical-guide-to-fine-tune-llms-with-lora-c835a99d7593, https://junbuml.ee/grad-flow-lora-grad-ckpt
    student_model.enable_input_require_grads()
    student_model.gradient_checkpointing_enable()

    return teacher_model, student_model, teacher_processor