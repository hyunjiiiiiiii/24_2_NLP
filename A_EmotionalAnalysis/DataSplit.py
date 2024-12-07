import torch
from sklearn.utils import shuffle

# 데이터 로드
train_encodings, train_labels = torch.load("C:/Users/CBNU/Desktop/이현지/24_2_NLP/Data/A_Data/train_data.pt")

# 데이터 셔플링
input_ids, attention_mask, labels = shuffle(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    train_labels,
    random_state=42
)

# 데이터 크기 확인
total_size = len(labels)
print(f"Total samples after shuffling: {total_size}")

# 데이터 분할 크기 설정
N = 4
split_size = total_size // N

for i in range(N):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size if i < N - 1 else total_size

    # 데이터 분할
    part_encodings = {
        "input_ids": input_ids[start_idx:end_idx],
        "attention_mask": attention_mask[start_idx:end_idx],
    }
    part_labels = labels[start_idx:end_idx]

    # 분할된 데이터 저장
    save_path = f"C:/Users/CBNU/Desktop/이현지/24_2_NLP/Data/A_Data/train_data_part{i+1}.pt"
    torch.save((part_encodings, part_labels), save_path)
    print(f"Saved: {save_path}")
