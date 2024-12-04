import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 업로드
train_data = pd.read_csv("/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/train_data.csv")
val_data = pd.read_csv("/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/val_data.csv")

# 레이블 매핑
label_mapping = {"중립": 0, "긍정": 1, "부정": 2}
train_data["label"] = train_data["label"].map(label_mapping)
val_data["label"] = val_data["label"].map(label_mapping)

# Train 데이터에서 Test 데이터 분리
train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

# 데이터 저장
train_data.to_csv("/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/train_data.csv", index=False, encoding="utf-8-sig")
val_data.to_csv("/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/val_data.csv", index=False, encoding="utf-8-sig")
test_data.to_csv("/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/test_data.csv", index=False, encoding="utf-8-sig")

print("Train, Validation, and Test datasets saved successfully!")