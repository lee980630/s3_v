import argparse
import json
import os
from typing import List, Dict, Any

from datasets import Dataset

# 시스템 프롬프트와 사용자 프롬프트는 data_construct_pipeline에서 사용한 것과 동일하게 정의
#원래 vrag prompt 사용 
# USER_PROMPT = '''You are a search agent.
# You must conduct reasoning inside <think> and </think> every time you get new information. 
# After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. 
# Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. 
# You can search as many times as you want. 
# If you determine that no further external knowledge is needed, you must finish with <search_complete>true</search_complete>. 
# Otherwise, continue with <search> or <bbox> actions until you are ready to finish. Question: {question}'''


# 위의 코드는 아래의 한 줄짜리 코드와 완전히 동일하게 동작합니다.
SYSTEM_PROMPT = """You are a search agent.\n
You must always begin with <think>...</think> showing your reasoning about the question.\n
After reasoning, output exactly one action tag among <search>...</search> or <bbox>[x1, y1, x2, y2]</bbox> or <search_complete>true</search_complete>.\n
Do not write anything before <think>. Keep actions on a new line after </think>.\n
When using <search>, vary or refine the query using evidence from previous steps, and do not repeat the same query twice."""


USER_QUESTION_FMT = "Question: {question}"


# all_examples = [example for example in all_examples if example['query'] not in sft_questions]
def convert_dataset(USER_PROMPT,file_list,file_source_list,output_name):
    all_examples = []
    for file_name, source_type in zip(file_list, file_source_list):
        with open(file_name, "r") as f:
            file_data = json.load(f)
            data_list = file_data["examples"]
            for item in data_list:
                item['source'] = source_type
            all_examples.extend(data_list)

    # for example in all_examples:
    #     if example['source'] == 'vidoseek':
    #         example['reason_type'] = example['meta_info']['query_type']
    #         example['content_type'] = example['meta_info']['source_type']
    #     elif example['source'] == 'slidevqa_test':
    #         query_type = example['meta_info']['query_type']
    #         if 'Multi-Hop' in query_type:
    #             example['reason_type'] = 'MultiHop'
    #         elif 'Single-Hop' in query_type:
    #             example['reason_type'] = 'SingleHop'
    #         if 'Non-Span' in query_type:
    #             example['content_type'] = 'NonSpan'
    #         elif 'Single-Span' in query_type:
    #             example['content_type'] = 'SingleSpan'
    #         elif 'Multi-Span' in query_type:
    #             example['content_type'] = 'MultiSpan'
    #     elif example['source'] == 'mmlongdoc':
    #         example['content_type'] = '####'.join(example['meta_info']['source_type'])
    #         example['reason_type'] = example['meta_info']['doc_type']
    #     else:
    #         example['content_type'] = 'Nan'
    #         example['reason_type'] = 'Nan'

    dataset = Dataset.from_dict({
        "id": [str(example["uid"]) for example in all_examples],
        "problem": [example["query"] for example in all_examples],
        #"prompt": [USER_PROMPT.replace('{question}',example["query"]) for example in all_examples],
        "answer": [example["reference_answer"] for example in all_examples],
        "file_name": [example["meta_info"]["file_name"] for example in all_examples],
        "reference_page": [example["meta_info"]["reference_page"] for example in all_examples],
        "data_source_type": [example["source"] for example in all_examples],
        #"query_content_type": [example["content_type"] for example in all_examples],
        #"query_reason_type": [example["reason_type"] for example in all_examples]
    })

    def make_map_fn_test(split):
        def process_fn(example, idx):
            #prompt = example.pop('prompt')
            answer = example.pop('answer')
            problem = example.pop('problem')
            data_source = example.pop('data_source_type')
            reference_page = example.pop('reference_page')
            file_name = example.pop('file_name')
            # images = example.pop('images')
            #query_content_type = example.pop('query_content_type')
            #query_reason_type = example.pop('query_reason_type')
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_QUESTION_FMT.format(question=problem)}
            ]

            data = {
                "data_source": data_source,
                "prompt": messages,
                # "images": images,
                #"ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": problem,
                    #"content_type": query_content_type,
                    #"reason_type": query_reason_type,
                    "file_name": file_name,
                    "reference_page": reference_page
                }
            }
            
            
            data['file_name'] = file_name
            data['reference_page'] = reference_page
            
            return data
        return process_fn

    # .map() 함수는 process_fn이 반환하는 새로운 구조의 데이터셋을 생성합니다.
    # 기존 'id', 'problem' 등의 열은 .pop()으로 인해 사라지고, 'data' 딕셔너리에 정의된 키들만 남게 됩니다.
    test_dataset = dataset.map(function=make_map_fn_test('test'), with_indices=True, num_proc=8)

    test_dataset.to_parquet(f'./{output_name}.parquet')


if __name__ == '__main__':
    convert_dataset(
        SYSTEM_PROMPT,       
        ['./dataset_score0.json'],#rl_train_dataset.json
        ['slidevqa_train'], #file source list
        'slidevqa_train_crop' #output name
    )
