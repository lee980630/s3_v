import pandas as pd

# Parquet 파일을 DataFrame으로 읽어오기
df = pd.read_parquet('./lsm_tmp/results/slidevqa_train_crop.parquet')

# 데이터 내용 확인
print(df.head())