import argparse
import json
import os
import random
from typing import Dict, List

import pandas as pd

random.seed(42)

PROMPT_INST = '''Answer the given question. 
You must conduct reasoning inside <think> and </think> every time you get new information. 
After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. 
You can search as many times as you want. If you determine that no further external knowledge is needed, you must finish with <search_complete>true</search_complete>. 
Otherwise, continue with <search> or <bbox> actions until you are ready to finish.'''

PROMPT_USER_START = "{question}"



def load_examples(path: str) -> List[Dict[str, str]]:
    """
    JSONL 파일을 로드하여 SFT 학습 형식에 맞는 prompt, response, image_path로 변환합니다.
    (모든 trajectory 정보를 포함하도록 수정됨)
    """
    examples = []
    image_base_path = "./corpus/img"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)

            question = data.get("query", "")
            prompt = f"assistant:\n{PROMPT_INST}\n\nuser:\n{question}"
            file_name = data.get("meta_info", {}).get("file_name", "")
            image_path = os.path.join(image_base_path,file_name)
            
            history = data.get("history", [])
            trajectory_parts = []
            
            for step in history:
                # ✨ 수정 1: if/elif -> 독립적인 if 문으로 변경
                # ✨ 수정 2: 누락된 정보(image, text, description) 처리 로직 추가
                
                if "think" in step:
                    trajectory_parts.append(f"<think>{step['think']}</think>")
                
                if "search" in step:
                    trajectory_parts.append(f"<search>{step['search']}</search>")

                # -- 관찰(Observation) 정보 처리 --
                if "image" in step:
                    # 관찰된 이미지 파일 이름을 텍스트로 명시해줄 수 있음
                    trajectory_parts.append(f"<image_observed>{step['image']}</image_observed>")

                if "text" in step:
                    trajectory_parts.append(f"<text_observed>{step['text']}</text_observed>")

                if "bbox" in step and step["bbox"]:
                    bbox_coords = next(iter(step["bbox"].values()), [])
                    trajectory_parts.append(f"<region>{bbox_coords}</region>")

                if "description" in step:
                    trajectory_parts.append(f"<description_observed>{step['description']}</description_observed>")
                # ------------------------------------

                if "search_complete" in step:
                    trajectory_parts.append(f"<search_complete>{str(step['search_complete']).lower()}</search_complete>")
            
            response = "".join(trajectory_parts)
            
            examples.append({
                "prompt": prompt, 
                "response": response, 
                "image_path": image_path
            })
            
    return examples

def split_dataset(data: List[Dict[str, str]], test_ratio: float = 0.2):
    """데이터를 섞고 훈련/검증 세트로 분할합니다."""
    random.shuffle(data)
    # split_idx = int(len(data) * (1 - test_ratio)) # 원본 분할 로직
    # 현재는 8개를 훈련용, 나머지를 테스트용으로 고정
    train_data = data[:8]
    test_data = data[8:]
    return train_data, test_data


def save_parquet(data: List[Dict[str, str]], path: str):
    """데이터를 Parquet 파일로 저장합니다."""
    # 경로가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(data).to_parquet(path, index=False)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Convert cot jsonl to train/test parquet files for SFT")
    # 사용자의 cot_crop.jsonl 파일이 있는 경로를 기본값으로 설정
    parser.add_argument("--input", default="./lsm_tmp/cot_crop.jsonl", help="Input JSONL file") 
    parser.add_argument("--output_dir", default="./lsm_tmp/results/sft_dataset", help="Directory to save parquet files")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Proportion of dataset used for test set (currently overridden)")
    args = parser.parse_args()

    print(f"Loading examples from: {args.input}")
    examples = load_examples(args.input)
    
    # 변환된 데이터 샘플 중 첫 번째 항목을 출력하여 확인
    if examples:
        print("\n--- Sample of converted data ---")
        print(f"Prompt: {examples[0]['prompt']}")
        print(f"Response: {examples[0]['response']}")
        print("--------------------------------\n")
        
    train_data, test_data = split_dataset(examples, test_ratio=args.test_ratio)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "val.parquet")

    print(f"Saving {len(train_data)} training examples to {train_path}")
    save_parquet(train_data, train_path)

    print(f"Saving {len(test_data)} validation examples to {test_path}")
    save_parquet(test_data, test_path)
    
    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    # 이 스크립트를 실행하기 전에 cot_crop.jsonl 파일이 동일한 디렉토리에 있는지 확인하세요.
    # 또는 --input 인자로 파일 경로를 지정하세요.
    # 예: python your_script_name.py --input /path/to/cot_crop.jsonl
    main()