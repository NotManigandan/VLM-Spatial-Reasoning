import re
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_question(raw_text):
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
        formatted.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["output"]}],
            }
        )
    return formatted


def collate_fn(examples, processor):
    texts = [processor.apply_chat_template(sample, tokenize=False) for sample in examples]
    image_batches = [process_vision_info(sample)[0] for sample in examples]
    batch = processor(text=texts, images=image_batches, return_tensors="pt", padding=True)
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


def prepare_datasets(cfg):
    print(f"Loading dataset: {cfg.dataset_id}:")
    raw_train = load_dataset(cfg.dataset_id, split="train")
    raw_eval = load_dataset(cfg.dataset_id, split="test")
    print("Formatting train samples:")
    train_ds = [format_data_spacethinker(s) for s in tqdm(raw_train, desc="Train")]
    print("Formatting eval samples:")
    eval_ds = [format_data_spacethinker(s) for s in tqdm(raw_eval, desc="Evaluation")]
    return train_ds, eval_ds


def prepare_dataloaders(cfg, processor, train_dataset, eval_dataset = None, use_collate_fn = False):
    if use_collate_fn is not False:
        collate_fn_proc = lambda ex: collate_fn(ex, processor)
    else:
        collate_fn_proc = None
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, collate_fn = collate_fn_proc)
    if eval_dataset is not None:
        eval_loader = DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn = collate_fn_proc)
        return train_loader, eval_loader
    return train_loader