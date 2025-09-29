from flask import Flask, request, jsonify
import json

# Flask 웹 애플리케이션을 생성합니다.
app = Flask(__name__)

# '/search' 경로로 "POST" 요청이 오면 search 함수를 실행합니다.
@app.route('/search', methods=['POST'])
def search():
    """
    수정된 generation.py로부터 POST 방식으로 검색 요청을 받아
    가짜 결과를 반환하는 API 엔드포인트입니다.
    """
    print("--- 👽 Mock Search Engine (POST): 새로운 요청 수신! 👽 ---")
    
    # 1. POST 요청의 본문(Body)에 담겨 온 JSON 데이터를 파이썬 객체로 읽어옵니다.
    #    request.get_json()은 'Content-Type: application/json' 헤더가 있는
    #    요청을 자동으로 파싱해줍니다.
    request_list = request.get_json()
    
    # 데이터가 없는 경우 에러를 반환합니다.
    if not request_list:
        print("  [경고] 요청 본문에 JSON 데이터가 없습니다.")
        return jsonify({"error": "Missing JSON in request body"}), 400
    
    print(f"  수신된 요청 리스트 ({len(request_list)}개): {request_list}")

    # 2. 각 요청에서 'query'를 추출하여 가짜 검색 결과를 만듭니다.
    dummy_results = [f"Result for '{req['query']}'" for req in request_list]
    
    print(f"  총 {len(dummy_results)}개의 더미 결과를 생성하여 반환합니다.")
    print("------------------------------------------------------\n")
    
    # 3. 'generation.py'로 결과 리스트를 JSON 형태로 반환합니다.
    return jsonify(dummy_results)

if __name__ == '__main__':
    # 이 파일을 직접 실행하면 Flask 개발 서버가 5000번 포트에서 시작됩니다.
    app.run(host='127.0.0.1', port=5000, debug=True)

