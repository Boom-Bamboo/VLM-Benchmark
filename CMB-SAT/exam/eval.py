import os
import torch
import platform
import subprocess
import json
from tqdm import tqdm
import re
import argparse
from chat_model import *
# from accelerate import Accelerator



def init_model():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    args=argparse.Namespace(fp16=True, skip_init=True, use_gpu_initialization=True, device='cuda')
    model = ChatModel(args)
    model = model.eval()
    return model, tokenizer

def main():
    model, tokenizer = init_model()
    
    with open('CMB-test-choice-question-merge.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    results = []
    for i in tqdm(range(len(data))):
        test_id = data[i]['id']
        exam_type = data[i]['exam_type']
        exam_class = data[i]['exam_class']
        question_type = data[i]['question_type']
        question = data[i]['question']
        
        F_opt = '\nF. ' + data[i]['option']['F'] if 'F' in data[i]['option'] else ''

        option_str = 'A. ' + data[i]['option']['A'] + '\nB. ' + data[i]['option']['B']+ '\nC. ' + data[i]['option']['C']+ '\nD. ' + data[i]['option']['D'] + '\nE. ' + data[i]['option']['E'] + F_opt
        
        prompt = f'以下是中国{exam_type}中{exam_class}考试的一道{question_type}。只回答正确选项，不要对正确选项进行任何解释和分析。\n题目：\n{question}\n选项：\n{option_str}'

        response, history = model.chat(tokenizer, prompt, history=[])
        matches = re.findall("[ABCDEF]", response)

        if matches == []:
            final_matches=""
        else:
            if question_type == "单项选择题":
                final_matches = matches[0]
            else:
                first_occurrence = {}
                final_matches = ""
                for char in matches:
                    if char in first_occurrence:
                        break
                    else:
                        final_matches += char
                        first_occurrence[char] = True

        final_result = "".join(final_matches)
        
        info = {
                "id": test_id,
                "model_answer": final_result
            }
        results.append(info)

    with open('output.json', 'w', encoding="utf-8") as f1:
        json.dump(results, f1, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
