# mock_search_engine.py

from flask import Flask, request, jsonify
import json

# Flask 애플리케이션 생성
app = Flask(__name__)

# '/search' 경로로 GET 요청이 오면 이 함수가 실행됨
@app.route('/search', methods=['GET'])
def search():
    print("--- 👽 새로운 요청 수신! 👽 ---")
    
    # URL 파라미터에서 'requests' 값을 가져옴
    # 예: ?requests=[{"query": "...", "uid": "..."}]
    requests_str = request.args.get('requests')
    
    print(f"수신된 원본 파라미터 문자열:\n{requests_str}\n")
    
    if requests_str:
        try:
            # 수신된 문자열을 파이썬 리스트/딕셔너리로 파싱
            parsed_data = json.loads(requests_str)
            print("성공적으로 파싱된 데이터:")
            # 예쁘게 출력
            print(json.dumps(parsed_data, indent=2, ensure_ascii=False))
            
            # 실제 검색 엔진인 것처럼, 가짜 결과 데이터를 반환해줘야 합니다.
            # 요청받은 쿼리 개수만큼 빈 결과를 생성해서 반환합니다.
            num_queries = len(parsed_data)
            dummy_results = [""] * num_queries # 빈 문자열 리스트
            
            return jsonify(dummy_results)

        except json.JSONDecodeError:
            print("오류: 수신된 문자열을 JSON으로 파싱할 수 없습니다.")
            return jsonify({"error": "Invalid JSON format"}), 400
    else:
        print("경고: 'requests' 파라미터가 요청에 포함되지 않았습니다.")
        return jsonify({"error": "'requests' parameter is missing"}), 400

if __name__ == '__main__':
    # 서버 실행 (IP: 127.0.0.1, 포트: 5000)
    app.run(host='127.0.0.1', port=5000, debug=True)