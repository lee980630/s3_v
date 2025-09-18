import pandas as pd

# ===== Pandas 출력 옵션 설정 (이 부분을 추가) =====
# 보여주는 최대 행 수를 None으로 설정하여 전체 행 출력
pd.set_option('display.max_rows', None)
# 보여주는 최대 열 수를 None으로 설정하여 전체 열 출력
pd.set_option('display.max_columns', None)
# 각 열의 최대 너비를 None으로 설정하여 내용이 잘리지 않게 함
pd.set_option('display.max_colwidth', None)
# 전체 출력 너비를 넓혀서 줄바꿈 방지
pd.set_option('display.width', 1000)
# =================================================

# 셸 스크립트에 지정된 학습 데이터 파일 경로
file_path = './data/rag/slidevqa_train_crop.parquet'

# 파일을 열어서 상위 5개 행과 컬럼 목록을 출력
try:
    df = pd.read_parquet(file_path)
    print("===== 파일 컬럼 목록 =====")
    print(df.columns)
    print("\n===== 데이터 샘플 (상위 5개) =====")
    # df.head() 대신 df 전체를 출력하거나, 보고 싶은 행의 개수를 늘립니다.
    # 예: 상위 20개 행을 보고 싶을 경우
    print(df.head(20))
except Exception as e:
    print(f"파일을 읽는 중 오류 발생: {e}")