import torch
from sklearn.utils import shuffle

# 데이터 로드
train_encodings, train_labels = torch.load("train_data.pt")

# 데이터 셔플링
train_encodings["input_ids"], train_encodings["attention_mask"], train_labels = shuffle(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    train_labels,
    random_state=42
)

# N등분으로 나누기
N = 4
split_size = len(train_encodings["input_ids"]) // N

for i in range(N):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size if i < N - 1 else len(train_encodings["input_ids"])

    # 데이터 분할
    part_encodings = {
        "input_ids": train_encodings["input_ids"][start_idx:end_idx],
        "attention_mask": train_encodings["attention_mask"][start_idx:end_idx]
    }
    part_labels = train_labels[start_idx:end_idx]

    # 분할된 데이터 저장
    torch.save((part_encodings, part_labels), f"train_data_part{i+1}.pt")
    print(f"train_data_part{i+1}.pt saved!")
