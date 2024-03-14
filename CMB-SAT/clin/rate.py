import json 
import openai
import re
import time
from tqdm import tqdm

openai.api_key = "***"
openai.api_type = "***"
openai.api_base = "***"
openai.api_version = "***"

background = f"您是一位AI评估员，专门评估其他语言模型提供的回答质量。您的目标是根据模型回答的流畅性、相关性、完整性和医学知识专业性四个标准对模型回答进行评分。请使用以下测评细则对每个标准进行评估\n流畅性：\n1分：完全破碎且无法阅读的句子片段\n2分：大部分破碎，只有少量可读的词\n3分：有一定流利度，但词汇有限\n4分：在表达复杂主题方面基本上是流畅的\n5分：人类水平流利度\n\n相关性：\n1分：与问题完全无关\n2分：与问题有一定关系，但主要是离题的\n3分：相关，但缺乏重点或关键细节\n4分：高度相关，解决了大部分问题\n5分：直接相关，准确地针对了问题\n\n完整性：\n1分：极度不完整\n2分：几乎不完整，信息有限\n3分：有一定的完整性，有一些信息\n4分：大部分信息都已显示\n5分：所有信息都已呈现\n\n医学知识专业性：\n1分：使用简单明了的语言，没有医学术语\n2分：具有一些医学知识，但缺乏深入细节\n3分：清晰地传达了一定的复杂医学信息\n4分：对医学术语有扎实的认识，但有些细节错误\n5分：在所有呈现的医学知识上都是完全正确的\n"
rate = f"根据测评细则，给出流畅性、相关性、完整性和医学知识专业性的评分。NO COMMENTS, NO ACKNOWLEDGEMENTS."

output_format = """
以JSON格式输出。
1. fluency字段的取值为float类型，表示模型回答的流畅性分数，保留小数点后一位
2. relevance字段的取值为float类型，表示模型回答的相关性分数，保留小数点后一位
3. completeness字段的取值为float类型，表示模型回答的完整性分数，保留小数点后一位
4. proficiency字段的取值为float类型，表示模型回答的医学知识专业性分数，保留小数点后一位
"""

with open('output.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
results = []

for i in tqdm(range(len(data))):
    eval_id = data[i]['id']
    description = data[i]['description']
    qa_pairs = data[i]['QA_pairs']
    responses = []
    fluency_ratings = 0.0
    relevance_ratings = 0.0
    completeness_ratings = 0.0
    proficiency_ratings = 0.0

    for j in tqdm(range(len(qa_pairs))):
        question = qa_pairs[j]['question']
        solution = qa_pairs[j]['answer']
        model_answer = qa_pairs[j]['model_answer']
        analyze = f"您将获得以下信息：\n\n问题（模型需要回答的问题）：\n{question}\n\n参考答案（该题目的参考答案，作为对模型回答打分的依据）：\n{solution}\n\n模型回答（测评对象，其他模型根据问题给出的回答）：\n{model_answer}\n\n"
    
        response = openai.ChatCompletion.create(
            engine = "gpt-4",
            messages = [
                {"role": "system", "content": background},
                {"role": "user", "content": rate + analyze + output_format}
            ],
            max_tokens = 500
        )
        if response['choices'][0]['message']['content'] == None:
            fluency_rating = 1.0
            relevance_rating = 1.0
            completeness_rating = 1.0
            proficiency_rating = 1.0
    
        else:
            ratings = response['choices'][0]['message']['content']
    
        ratings = json.loads(ratings)
        fluency_rating = ratings['fluency']
        relevance_rating = ratings['relevance']
        completeness_rating = ratings['completeness']
        proficiency_rating = ratings['proficiency']
        
        fluency_ratings = fluency_rating + fluency_ratings
        relevance_ratings = relevance_rating + relevance_ratings
        completeness_ratings = completeness_rating + completeness_ratings
        proficiency_ratings = proficiency_rating + proficiency_ratings

    fluency_ratings = round(fluency_ratings/len(qa_pairs),1)
    relevance_ratings = round(relevance_ratings/len(qa_pairs),1)
    completeness_ratings = round(completeness_ratings/len(qa_pairs),1)
    proficiency_ratings = round(proficiency_ratings/len(qa_pairs),1)

    info = {
            "id": eval_id,
            "fluency": fluency_ratings,
            "relevance": relevance_ratings,
            "completeness": completeness_ratings,
            "proficiency": proficiency_ratings
        }
    results.append(info)

with open('ratings.json', 'w', encoding="utf-8") as f1:
    json.dump(results, f1, ensure_ascii=False, indent=4)
