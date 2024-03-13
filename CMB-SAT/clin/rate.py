import json 
import openai
import re
from tqdm import tqdm

openai.api_key = "***"
openai.api_type = "***"
openai.api_base = "***"
openai.api_version = "***"

background = f"You are an AI evaluator specializing in assessing the quality of answers provided by other language models . Your primary goal is to rate the answers based on their fluency , relevance , completeness , proficiency in medicine . Use the following scales to evaluate each criterion :\nFluency :\n1: Completely broken and unreadable sentence pieces\n2: Mostly broken with few readable tokens\n3: Moderately fluent but with limited vocabulary\n4: Mostly coherent in expressing complex subjects\n5: Human - level fluency\n\nRelevance :\n1: Completely unrelated to the question\n2: Some relation to the question , but mostly off - topic\n3: Relevant , but lacking focus or key details\n4: Highly relevant , addressing the main aspects of the question\n5: Directly relevant and precisely targeted to the question\n\nCompleteness :\n1: Extremely incomplete\n2: Almost incomplete with limited information\n3: Moderate completeness with some information\n4: Mostly complete with most of the information displayed\n5: Fully complete with all information presented\n\nProficiency in medicine :\n1: Using plain languages with no medical terminology .\n2: Equipped with some medical knowledge but lacking in - depth details\n3: Conveying moderately complex medical information with clarity\n4: Showing solid grasp of medical terminology but having some minor mistakes in detail\n5: Fully correct in all presented medical knowledge\n"
rate = [
    "You directly give the ratings for fluency, with ratings ranging from 1-5. There is no need to analyze and interpret the ratings.",
    "You directly give the ratings for relevance, with ratings ranging from 1-5. There is no need to analyze and interpret the ratings.",
    "You directly give the ratings for completeness, with ratings ranging from 1-5. There is no need to analyze and interpret the ratings.",
    "You directly give the ratings for proficiency in medicine, with ratings ranging from 1-5. There is no need to analyze and interpret the ratings."
]

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
        analyze = f"You are provided with the following information :\n\n- a description:\n[description]\n{description}\n[end of description]\n\n- a question based on the description:\n[question]\n{question}\n[end of question]\n\n- the solution to the question:\n[solution]\n{solution}\n[end of solution]\n\n- the model's answer to the question:\n[model's answer]\n{model_answer}\n[end of model's answer]\n"

        response = openai.ChatCompletion.create(
            engine = "gpt-4",
            messages = [
                {"role": "system", "content": background},
                {"role": "user", "content": analyze + rate[0]}
            ],
            max_tokens = 50
        )
        fluency = response['choices'][0]['message']['content']
        match = re.search(r'-?\d+(\.\d+)?', fluency)
        if match == None:
            fluency_rating = 0.0
        else:
            fluency_rating = float(match.group())
        fluency_ratings = fluency_rating + fluency_ratings


        response = openai.ChatCompletion.create(
            engine = "gpt-4",
            messages = [
                {"role": "system", "content": background},
                {"role": "user", "content": analyze + rate[1]}
            ],
            max_tokens = 50
        )
        relevance = response['choices'][0]['message']['content']
        match = re.search(r'-?\d+(\.\d+)?', relevance)
        if match == None:
            relevance_rating = 0.0
        else:
            relevance_rating = float(match.group())
        relevance_ratings = relevance_rating + relevance_ratings


        response = openai.ChatCompletion.create(
            engine = "gpt-4",
            messages = [
                {"role": "system", "content": background},
                {"role": "user", "content": analyze + rate[2]}
            ],
            max_tokens = 50
        )
        completeness = response['choices'][0]['message']['content']
        match = re.search(r'-?\d+(\.\d+)?', completeness)
        if match == None:
            completeness_rating = 0.0
        else:
            completeness_rating = float(match.group())
        completeness_ratings = completeness_rating + completeness_ratings


        response = openai.ChatCompletion.create(
            engine = "gpt-4",
            messages = [
                {"role": "system", "content": background},
                {"role": "user", "content": analyze + rate[3]}
            ],
            max_tokens = 50
        )
        proficiency = response['choices'][0]['message']['content']
        match = re.search(r'-?\d+(\.\d+)?', proficiency)
        if match == None:
            proficiency_rating = 0.0
        else:
            proficiency_rating = float(match.group())
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

