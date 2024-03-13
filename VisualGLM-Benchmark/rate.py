import json 
import openai
import re
from tqdm import tqdm

openai.api_key = "***"
openai.api_type = "***"
openai.api_base = "***"
openai.api_version = "***"

background = f"You are an AI evaluator specializing in assessing the quality of answers provided by other language models . Your primary goal is to rate the answers based on their fluency , relevance , completeness , proficiency in medicine . Use the following scales to evaluate each criterion :\nFluency :\n1: Completely broken and unreadable sentence pieces\n2: Mostly broken with few readable tokens\n3: Moderately fluent but with limited vocabulary\n4: Mostly coherent in expressing complex subjects\n5: Human - level fluency\n\nRelevance :\n1: Completely unrelated to the question\n2: Some relation to the question , but mostly off - topic\n3: Relevant , but lacking focus or key details\n4: Highly relevant , addressing the main aspects of the question\n5: Directly relevant and precisely targeted to the question\n\nCompleteness :\n1: Extremely incomplete\n2: Almost incomplete with limited information\n3: Moderate completeness with some information\n4: Mostly complete with most of the information displayed\n5: Fully complete with all information presented\n\nProficiency in medicine :\n1: Using plain languages with no medical terminology .\n2: Equipped with some medical knowledge but lacking in - depth details\n3: Conveying moderately complex medical information with clarity\n4: Showing solid grasp of medical terminology but having some minor mistakes in detail\n5: Fully correct in all presented medical knowledge\n"
rate = f"Based on the scales, You directly give the ratings for fluency, relevance, completeness, and proficiency in medicine."

with open('output.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
results = []

for i in tqdm(range(len(data))):
    image = data[i]['image']
    solution = data[i]['standard_answer']
    model_answer = data[i]['model_answer']
    question = data[i]['prompt']
    
    analyze = f"You are provided with the following information:\n\n- a question:\n[question]\n{question}\n[end of question]\n\n- the standard answer to the question:\n[standard answer]\n{solution}\n[end of standard answer]\n\n- the model's answer to the question:\n[model's answer]\n{model_answer}\n[end of model's answer]\n"
    
    response = openai.ChatCompletion.create(
        engine = "gpt-4",
        messages = [
            {"role": "system", "content": background},
            {"role": "user", "content": analyze + rate}
        ],
        max_tokens = 500
    )
    ratings = response['choices'][0]['message']['content']
    match = re.findall(r'\d+', ratings)
    fluency_rating = float(match[0])
    relevance_rating = float(match[1])
    completeness_rating = float(match[2])
    proficiency_rating = float(match[3])

    info = {
            "image": image,
            "fluency": fluency_rating,
            "relevance": relevance_rating,
            "completeness": completeness_rating,
            "proficiency": proficiency_rating
        }
    results.append(info)

with open('ratings.json', 'w', encoding="utf-8") as f1:
    json.dump(results, f1, ensure_ascii=False, indent=4)
