import requests
import time
import json

# 스크립트에 설정된 search_url 주소입니다.
SEARCH_URL = "http://0.0.0.0:8004/search"

# 실제 Retriever가 받을 만한 간단한 테스트 데이터입니다.
# 실제 데이터 형식에 맞게 수정하시면 더 정확합니다.
dummy_payload = {
    "queries": ["What is the capital of South Korea?"]
}

request_times = []
num_tests = 20 # 20번 테스트해서 평균을 내보겠습니다.

print(f"'{SEARCH_URL}'에 테스트 요청을 시작합니다...")

for i in range(num_tests):
    try:
        # 1. 요청 직전 시간 기록
        start_time = time.time()

        # 2. 실제 POST 요청 보내기
        response = requests.get(SEARCH_URL, params=dummy_payload)
        response.raise_for_status() # HTTP 에러가 있으면 예외 발생

        # 3. 응답 받은 직후 시간 기록
        end_time = time.time()

        # 4. 소요 시간 계산 (밀리초 단위로 변환)
        duration_ms = (end_time - start_time) * 1000
        request_times.append(duration_ms)

        print(f"요청 #{i+1}: {duration_ms:.2f} ms")

    except requests.exceptions.RequestException as e:
        print(f"요청 #{i+1} 실패: {e}")
        break

# 성공한 요청들의 평균 시간 계산
if request_times:
    average_time = sum(request_times) / len(request_times)
    print("\n--- 결과 ---")
    print(f"총 {len(request_times)}번의 성공적인 요청")
    print(f"평균 응답 시간: {average_time:.2f} ms")