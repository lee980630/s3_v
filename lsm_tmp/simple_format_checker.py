import re
import json
import os
from datetime import datetime

# --- 로그 기능 추가 ---
# 로그를 저장할 디렉토리와 파일 경로를 설정합니다.
LOG_DIR = "./checker_logs"
LOG_FILE = os.path.join(LOG_DIR, "solution_str_log.txt")

# 스크립트 실행 시 로그 디렉토리가 없으면 자동으로 생성합니다.
os.makedirs(LOG_DIR, exist_ok=True)
# --------------------

def simple_format_checker(data_source, solution_str, ground_truth, extra_info):
    """
    Assistant의 전체 대화 기록(solution_str)을 검사하여,
    모든 턴이 정해진 문법 규칙을 따랐는지 확인하고, 입력값을 로그로 남깁니다.
    """

    # --- 로그 기능 추가 ---
    # 'a' 모드(append)로 파일을 열어, 기존 로그에 새로운 내용을 추가합니다.
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"========== Log at {timestamp} ==========\n")
        f.write("Received solution_str:\n")
        f.write("-" * 40 + "\n")
        f.write(solution_str)
        f.write("\n" + "-" * 40 + "\n\n")
    # --------------------

    # 1단계: Assistant의 턴(Turn)만 분리하기
    assistant_turns = re.findall(r"<\|im_start\|>assistant(.*?)<\|im_end\|>", solution_str, re.DOTALL)

    if not assistant_turns:
        return 0.0

    # 2단계: 모든 턴에 대한 '공통 문법' 검사
    for i, turn in enumerate(assistant_turns):
        cleaned_turn = turn.strip()

        # 2-A: <think> 태그 검사
        if not cleaned_turn.startswith('<think>'):
            return 0.0
        if cleaned_turn.count('<think>') != 1 or cleaned_turn.count('</think>') != 1:
            return 0.0

        # 2-B: 행동(Action) 태그 개수 검사
        action_count = cleaned_turn.count('<search>') + cleaned_turn.count('<bbox>') + cleaned_turn.count('<search_complete>')
        if action_count != 1:
            return 0.0

        # 2-C: 행동(Action) 태그 내용 검사 (세부 규칙)
        if '<search>' in cleaned_turn:
            match = re.search(r"<search>(.*?)</search>", cleaned_turn, re.DOTALL)
            if not match or not match.group(1).strip():
                return 0.0

        elif '<bbox>' in cleaned_turn:
            match = re.search(r"<bbox>(.*?)</bbox>", cleaned_turn, re.DOTALL)
            if not match:
                return 0.0
            try:
                bbox_content = json.loads(match.group(1).strip())
                if not isinstance(bbox_content, list) or len(bbox_content) != 4:
                    return 0.0
                if not all(isinstance(coord, (int, float)) for coord in bbox_content):
                    return 0.0
            except json.JSONDecodeError:
                return 0.0

        elif '<search_complete>' in cleaned_turn:
            if '<search_complete>true</search_complete>' not in cleaned_turn.replace(" ", ""):
                return 0.0

    # 3단계: '마지막 턴' 특별 규칙 검사
    last_turn = assistant_turns[-1].strip()
    if '<search_complete>' not in last_turn:
        return 0.0

    # 4단계: 최종 합격 판정
    return 1.0