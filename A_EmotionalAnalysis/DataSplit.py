import pandas as pd

# 원본 CSV 파일 경로
csv_path = "/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/train_data.csv"

# 데이터 로드
data = pd.read_csv(csv_path)

# 데이터 섞기 (선택 사항)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 데이터 나누기
N = 3
chunk_size = len(data) // N

for i in range(N):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i < N - 1 else len(data)

    chunk = data[start_idx:end_idx]
    chunk.to_csv(f"train_data_part{i+1}.csv", index=False, encoding="utf-8-sig")
    print(f"Saved: train_data_part{i+1}.csv")
