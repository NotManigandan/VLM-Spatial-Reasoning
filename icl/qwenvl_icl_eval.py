import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoTokenizer
from utils import SYS_PROMPTS, FORMAT_PROMPTS, make_client, llm_judge
from peft import PeftModel
##### OUR IMPLEMENTATION STARTS#####
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
import time
import numpy as np 
import pickle
from datetime import datetime
import csv
import pandas as pd

VQA_DATASET = "remyxai/SpaceThinker"

# ref: from train.py
def extract_question(raw_text: str) -> str:
    pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>"
    m = re.search(pattern, raw_text, re.DOTALL)
    return m.group(1).strip() if m else raw_text.strip()

DATASET = "remyxai/SpaceThinker"
TRAIN_VQA = load_dataset(VQA_DATASET, split="train")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

QUESTIONS = []
IMAGES = []
ANSWERS = []
LATENCY = []
LOOKUP = {}

def create_image_lookup():
    global QUESTIONS, IMAGES, ANSWERS, LOOKUP
    for sample in tqdm(TRAIN_VQA, desc="Creating Image Lookup using Question as Key"):
        question = extract_question(sample.get("input", ""))    
        image = sample.get("images")
        LOOKUP[ question ] = image[0] or None
    return LOOKUP


def add_missing_images_and_fix_ans(df):
    global QUESTIONS, IMAGES, ANSWERS, LOOKUP
    for row in df.itertuples(index=False):
        q = row.question
        IMAGES.append( LOOKUP[ q ] )
        a = row.answers
        parts = a.split("</think>")
        reasoning = parts[ 0 ] + "</think>"
        ans = parts[ 1 ].strip()
        ANSWERS.append( f"{reasoning}\nAnswer: {ans}")
    return IMAGES, ANSWERS

def create_mcq_embeddings():
    global QUESTIONS, IMAGES, ANSWERS, LOOKUP
    df = pd.read_csv("SpaceThinkerMCQ.csv")
    LOOKUP = create_image_lookup()
    QUESTIONS = df["question_mcq"].tolist()
    add_missing_images_and_fix_ans(df)
    embeddings = embedding_model.encode(QUESTIONS)
    return embeddings


# grab questions, images, answers from SpaceThinker dataset which was used as training
# dataset for the SFT model. The ICL examples are picked from the same training dataset
# for fair comparability in performance for the ICL vs SFT model.
def create_embeddings():

    for sample in tqdm(TRAIN_VQA, desc="Creating Embedding from SpaceThinker VQA"):
        question = extract_question(sample.get("input", ""))    
        image = sample.get("images")
        answer = sample.get("output")
        if not question or not image or not answer:
            continue
        think_text = re.search(r"<think>(.*?)</think>", sample.get("output"), re.DOTALL)
        if not think_text:
            continue
        think_text = think_text.group(1).strip()
        ans_text = re.search(r"<answer>(.*?)</answer>", sample.get("output"), re.DOTALL)
        if not ans_text:
            continue
        ans_text = ans_text.group(1).strip()
        answer_with_reasoning = think_text + "\nAnswer : " + ans_text
        ANSWERS.append( answer_with_reasoning ) # remove the tags
        IMAGES.append( image[0] )
        QUESTIONS.append( question )

    embeddings = embedding_model.encode(QUESTIONS)
    return embeddings


# reference: https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/
def create_faiss_index():
    if args.useMCQ:
        embeddings = create_mcq_embeddings()
    else:
        embeddings = create_embeddings()
    N, dim = embeddings.shape 
    print( f"embedding shape: ({N}, {dim})" )
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print( index.ntotal )
    return index


def get_similar_questions_data( query, index, k=3 ):
    query_embedding = embedding_model.encode( query )
    query_embedding = query_embedding.reshape(1, -1) # faiss expects n,d shape for vector
    _, IDX = index.search( query_embedding, k )
    icl_samples = []
    for i in range( k ):
        idx = IDX[ 0 ][ i ]
        entry = {}
        entry[ "question"] = QUESTIONS[idx]
        entry[ "image" ] = IMAGES[idx]
        entry[ "answer"] = ANSWERS[idx]
        icl_samples.append( entry )
    return icl_samples


def add_icl_examples( messages, query, index, k=3 ):
    icl_samples = get_similar_questions_data( query, index, k )
    for sample in icl_samples:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": sample[ "image" ]},
                {"type": "text",  "text": sample[ "question" ]},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text",  "text": sample[ "answer" ]},
            ],
        })
    return messages

##### OUR IMPLEMENTATION ENDS #####

def blank_stats():
    return {
        "Total": [],
        "Dynamic_Reasoning": {"Manipulation": [], "Motion_Analysis": [], "Total": []},
        "Spatial_Interaction": {"Traffic_Analysis": [], "Localization": [], "Geospatial_Strategy": [], "Total": []},
        "Complex_Logic": {"Pattern_Recognition": [], "Geometric_Reasoning": [], "Total": []},
        "Perspective_Taking": {"Egocentric": [], "Allocentric": [], "Hypothetical": [], "Total": []},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B")
    # "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen2.5-VL-32B-Instruct" "Qwen/Qwen2.5-VL-72B-Instruct"
    # "remyxai/SpaceQwen2.5-VL-3B-Instruct" "remyxai/SpaceThinker-Qwen2.5VL-3B"
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--useMCQ", action="store_true")
    parser.add_argument("--model_adapters", choices=["/home/ubuntu/10623_dev_workspace/spacethinker-lora", ""], default="")
    parser.add_argument("--prompt_type", choices=["none", "zeroshot_cot", "manual_cot"], default="manual_cot")
    parser.add_argument("--eval_type", choices=["re", "json", "llm", "direct"], default="re")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--group_index", type=int, default=0) # parallel eval index of sub-process
    parser.add_argument("--group", type=int, default=1) # parallel eval number of sub-process
    # parallel eval parameters, if not use parallel eval, set group_index to 0 and group to 1
    parser.add_argument("--dataset_path", default="/home/ubuntu/OmniSpatial/dataset/OmniSpatial-test")
    parser.add_argument("--result_path", default="result")
    args = parser.parse_args()

    result_path = os.path.join(args.result_path, args.model_id)
    os.makedirs(result_path, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    output_file = os.path.join(result_path, f"results_{args.group_index}_k_{args.k}_{timestamp}.json")
    print('output_file: ', output_file)

    model_id = args.model_id
    model_adapters = args.model_adapters
    if model_adapters != "":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            quantization_config=bnb
        )
    
        model = PeftModel.from_pretrained(base_model, model_adapters)
        model = model.merge_and_unload()
    else:
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            load_in_8bit=False
        )
        model = base_model

    # processor = AutoProcessor.from_pretrained(model_id)
    # processor = AutoProcessor.from_pretrained(model_adapters)
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_adapters or model_id, use_fast=False)
    processor.tokenizer = tokenizer

    data = json.load(open(os.path.join(args.dataset_path, 'data.json'))) * args.repeat
    
    total = len(data)
    print("Full dataset size: ", total)
    print(f"Your group dataset size: {len(data)}")
    result = blank_stats()

    if args.eval_type == "llm":
        client = make_client()
    ##### OUR IMPLEMENTATION STARTS#####
    if args.k > 0:
        index = create_faiss_index()
    ##### OUR IMPLEMENTATION ENDS#####
    for info in tqdm(data):
        raw_id = info["id"]

        question = info["question"]
        options = info["options"]
        answer = info["answer"]
        task_type = info["task_type"]
        sub_task_type = info["sub_task_type"]

        ##### OUR IMPLEMENTATION STARTS#####
        # system prompt goes first
        prompt = SYS_PROMPTS[args.prompt_type] + '\n' + FORMAT_PROMPTS[args.eval_type]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if args.k > 0:
            # add the ICL examples with k=3
            messages = add_icl_examples( messages, question, index, k=args.k )


        # now add the evaluation question
        image_path = os.path.join(args.dataset_path, task_type, f"{raw_id.split('_')[0]}.png")
        prompt = "\n\n" + question
        for i in range(len(options)):
            prompt += f"\n{chr(65 + i)}. {options[i]}"
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        ##### OUR IMPLEMENTATION ENDS##### 
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        #import pdb; pdb.set_trace()
        # Inference: Generation of the output
        ##### OUR IMPLEMENTATION STARTS##### 
        start_time = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        end_time = time.time()
        delta_t = end_time - start_time
        LATENCY.append( delta_t )
        ##### OUR IMPLEMENTATION ENDS##### 
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
        #print(response)

        gt_letter = chr(65 + answer)
        if args.eval_type == "json":
            try:
                cleaned = response.strip().removeprefix("```json").removesuffix("```").strip()
                pred_letter = json.loads(cleaned).get("answer", "A").strip().upper()[:1]
            except Exception:
                pred_letter = "A"
            flag = pred_letter == gt_letter
        elif args.eval_type == "re":
            PATTERN = re.compile(r"Answer\s*:\s*([A-D])\b", re.IGNORECASE)
            pred_letter = PATTERN.findall(response)[-1] if len(PATTERN.findall(response)) > 0 else "A"
            flag = pred_letter == gt_letter
        elif args.eval_type == "direct":
            pred_letter = response.strip().upper()[:1]
            flag = pred_letter == gt_letter
        elif args.eval_type == "llm":
            flag = llm_judge(question=prompt, pred=response, gt=gt_letter, client=client, judge_model="gpt-4.1-mini")
        else:
            assert False, f"Unknown eval_type: {args.eval_type}"
        #import pdb; pdb.set_trace()
        result["Total"].append(flag)
        result[task_type][sub_task_type].append(flag)
        result[task_type]["Total"].append(flag)

    # final report -----------------------------------------------------------
    eps = 1e-6
    overall = sum(result["Total"]) / (len(result["Total"])+eps) * 100
    print("\n======= FINAL =======")
    print(f"Overall: {overall:.2f}% (N={len(result['Total'])})")
    for task in [k for k in result if k not in {"Total", "Processed"}]:
        task_acc = sum(result[task]["Total"]) / (len(result[task]["Total"])+eps) * 100
        print(f"{task}: {task_acc:.2f}%")
        for sub in result[task]:
            if sub == "Total":
                continue
            sub_acc = sum(result[task][sub]) / (len(result[task][sub])+eps) * 100
            print(f"    {sub}: {sub_acc:.2f}%")

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)

    print("\n======= LATENCY =======")
    print(f"Average latency: {np.mean(LATENCY):.3f} seconds")
    print(f"Max latency: {np.max(LATENCY):.3f} seconds")
    print(f"Min latency: {np.min(LATENCY):.3f} seconds")
    latency_path = os.path.join(result_path, f"latency_{args.group_index}_k_{args.k}_{timestamp}.pkl")
    with open(latency_path, "wb") as f:
        pickle.dump(LATENCY, f)
    print(f"Saved latency estimates at: {latency_path}")
