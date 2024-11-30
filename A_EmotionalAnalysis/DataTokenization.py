import torch
import pandas as pd
from transformers import AutoTokenizer

# 데이터 경로
train_data_path = "./Data/A_Data/train_data.csv"
val_data_path = "./Data/A_Data/val_data.csv"

# 데이터 로드
train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)

# KoBERT 토크나이저 로드
model_name = "monologg/kobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 토큰화 함수
def tokenize_data(data, tokenizer):
    return tokenizer(
        list(data["text"]),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

# 데이터 토큰화
train_encodings = tokenize_data(train_data, tokenizer)
val_encodings = tokenize_data(val_data, tokenizer)

# 레이블 텐서로 변환
train_labels = torch.tensor(train_data["label"].tolist(), dtype=torch.long)
val_labels = torch.tensor(val_data["label"].tolist(), dtype=torch.long)

# TensorDataset 생성 후 저장
torch.save((train_encodings, train_labels), "train_data.pt")
torch.save((val_encodings, val_labels), "val_data.pt")

print("Tokenized data saved as train_data.pt and val_data.pt")
