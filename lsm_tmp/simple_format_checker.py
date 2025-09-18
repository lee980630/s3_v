import json

def simple_format_checker(data_source, solution_str, ground_truth, extra_info):

    
    # #임시방편
    
    # """
    # score값 구현
    # 모델이 생성한 응답(solution_str)이 미리 정의된 JSON 형식을 따르는지 검사합니다.
    # - 올바른 형식이면 1.0을 반환합니다.
    # - 형식이 올바르지 않으면 0.0을 반환합니다.
    # """
    # # 1. JSON 형식 확인: 문자열을 파싱할 수 있는지 검사
    # try:
    #     data = json.loads(solution_str)
    #     if not isinstance(data, dict):
    #         # JSON이지만 딕셔너리가 아닌 경우 (예: "[1, 2]")
    #         print("response not a json")
    #         return 0.0
    # except json.JSONDecodeError:
    #     # JSON 형식 문법 오류
    #     return 0.0

    # # 2. 필수 키('think') 존재 여부 확인
    # if "think" not in data:
    #     return 0.0

    # # 3. 행동 유형별 구조 검증
    # # A. Search 행동 검증
    # if "search" in data:
    #     # 키 개수가 2개이고, 'search' 값이 비어있지 않은 문자열인지 확인
    #     if len(data) == 2 and isinstance(data.get("search"), str) and data["search"]:
    #         return 1.0

    # # B. Crop 행동 검증
    # elif "bbox" in data:
    #     bbox = data["bbox"]
    #     if len(data) == 2 and isinstance(bbox, list) and len(bbox) == 4 \
    #        and all(isinstance(coord, (int, float)) for coord in bbox):
    #         return 1.0


    # # C. 완료 행동 검증
    # elif "search_complete" in data:
    #     # 키 개수가 2개이고, 'search_complete' 값이 boolean True인지 확인
    #     if len(data) == 2 and data["search_complete"] is True:
    #         return 1.0

    # # 4. 예외 처리: 위 어떤 형식에도 해당하지 않으면 0.0 반환
    # #return 0.0
    return 1.0