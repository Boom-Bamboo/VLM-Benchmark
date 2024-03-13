import os
import torch
import platform
import subprocess
import json
from tqdm import tqdm
import re
import argparse
from chat_model import *



def init_model():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    args=argparse.Namespace(fp16=True, skip_init=True, use_gpu_initialization=True, device='cuda')
    model = ChatModel(args)
    model = model.eval()
    return model, tokenizer

def main():
    model, tokenizer = init_model()
    
    with open('CMB-Clin-qa.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    results = []
    for i in tqdm(range(len(data))):
        test_id = data[i]['id']
        clin_title = data[i]['title']
        clin_description = data[i]['description']
        qa_pairs = data[i]['QA_pairs']
        prompt0 = f'你是一名经验丰富的医生，现在你需要进行{clin_title}，针对我给出的问题进行回答。下面是对病情的描述：\n{clin_description}'
        response0, history = model.chat(tokenizer, prompt0, history=[])
        QAs = []
        for j in tqdm(range(len(qa_pairs))):
            clin_question = qa_pairs[j]['question']
            clin_answer = qa_pairs[j]['answer']
            prompt = f'问题{j}：{clin_question}'
            response, history = model.chat(tokenizer, prompt, history=history)
            QA = {
                "question": clin_question,
                "answer": clin_answer,    
                "model_answer": response
                }
            QAs.append(QA)
        info = {
                "id": test_id,
                "title": clin_title,
                "description": clin_description,
                "QA_pairs": QAs
            }

        results.append(info)

    with open('output.json', 'w', encoding="utf-8") as f1:
        json.dump(results, f1, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()

