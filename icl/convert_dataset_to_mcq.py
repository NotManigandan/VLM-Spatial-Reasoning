# convert the SpaceThinker dataset to an MCQ dataset.
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
import time
import numpy as np 
import pickle
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import ast
import random
import re
import csv
import pandas as pd

VQA_DATASET = "remyxai/SpaceThinker"
model_id = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
PHI_MODEL = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
DATASET = "remyxai/SpaceThinker"
TRAIN_VQA = load_dataset(VQA_DATASET, split="train")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_PROMPT = """
You are a helpful Assistant. Given the following question and correct answer pair,
generate 3 different incorrect, but plausible choices for the answer.

Question: {}
Answer: {}

Return only the 3 generated options as a PYTHON list, and nothing else.
"""

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = PHI_MODEL.generate( **inputs,
                             max_new_tokens=150,
                             temperature=0.6,
                             do_sample=True,
                             top_p=0.8)

    _, prompt_prefix_len =  inputs[ "input_ids" ].shape
    response = tokenizer.decode(output[0][ prompt_prefix_len: ], skip_special_tokens=True).strip()
    return response


# ref: from train.py
def extract_question(raw_text: str) -> str:
    pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>"
    m = re.search(pattern, raw_text, re.DOTALL)
    return m.group(1).strip() if m else raw_text.strip()


def create_mcq_qa( question, answer, options_as_list ):
    # make sure there are exactly 4 options in all
    options = options_as_list[ : 3 ] + [ answer ]
    letters = list( "ABCD" )[ : len( options ) ]
    labels = random.sample( letters, len( options ) )
    pairs = dict( zip( labels, options ) )
    answer_pair = None
    for ch, ans in pairs.items():
        if ans == answer:
            answer_pair = (ch, ans)
            break
    assert answer_pair is not None

    question_patch = ""
    for letter in letters:
        question_patch += f"{letter}. {pairs[letter]}\n"

    formatted_question = question + question_patch
    formatted_answer = "{}. {}".format( answer_pair[ 0 ], answer_pair[ 1 ] )
    return formatted_question, formatted_answer

def create_mcq_dataset():
    QUESTIONS = []
    MCQ_QUESTIONS = []
    IMAGES = []
    ANSWERS = []

    for sample in tqdm(TRAIN_VQA, desc="Creating MCQ VQA from SpaceThinker"):
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

        # Now we generate other options
        prompt = BASE_PROMPT.format(question, ans_text)
        response = generate_response( prompt )
        matching_list = re.search(r"\[[\s\S]*?\]", response)
        matches = re.findall(r"['\"](.*?)['\"]", response)
        if matches:
            # sometimes the optoins are  not properly enclosed between []
            options_as_list = matches
        else:
            # sometimes the model generates a response of numbered list instead of python list
            options_as_list = re.findall(r"^\s*\d+\.\s*(.+)", response, re.MULTILINE)
        #import pdb; pdb.set_trace();
        if not options_as_list:
            continue
        #print( options_as_list )

        # now create MCQ lables for correct answers + these 3 labels
        formatted_question, formatted_answer = create_mcq_qa( question, ans_text, options_as_list )
        # keeping the think tags, helps us select the answer more easily at inference 
        formatted_answer_with_reasoning = \
        """<think>{}</think>\n{}""".format(think_text, formatted_answer)
        ANSWERS.append( formatted_answer_with_reasoning ) # remove the tags
        IMAGES.append( image[0] )
        QUESTIONS.append( question )
        MCQ_QUESTIONS.append( formatted_question )

    return QUESTIONS, MCQ_QUESTIONS, ANSWERS, IMAGES


QUESTIONS, MCQ_QUESTIONS, ANSWERS, IMAGES = create_mcq_dataset()


df = pd.DataFrame({
    "question": QUESTIONS,
    "question_mcq": MCQ_QUESTIONS,
    "answers": ANSWERS
})

df.to_csv("SpaceThinkerMCQ.csv", index=True)


data = {
    "questions": QUESTIONS,
    "question_mcq": MCQ_QUESTIONS,
    "answers": ANSWERS,
    "images": IMAGES
}

with open("SpaceThinkerMCQ.pkl", "wb") as f:
    pickle.dump(data, f)