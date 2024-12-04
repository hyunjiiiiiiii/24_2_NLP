import pandas as pd

# 파일 경로 지정
train_file_1 = "/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/emotion_dataset_01.csv"
train_file_2 = "/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/emotion_dataset_01.csv"

# 파일 병합
train_data_1 = pd.read_csv(train_file_1)
train_data_2 = pd.read_csv(train_file_2)
train_data = pd.concat([train_data_1, train_data_2], ignore_index=True)

# 병합된 train 데이터 저장
train_data.to_csv("/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/train_data.csv", index=False, encoding="utf-8-sig")
print("Merged Train Data saved!")

# 파일 경로 지정
val_file_1 = "/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/emotion_vdataset_01.csv"
val_file_2 = "/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/emotion_vdataset_01.csv"

# 파일 병합
val_data_1 = pd.read_csv(val_file_1)
val_data_2 = pd.read_csv(val_file_2)
val_data = pd.concat([val_data_1, val_data_2], ignore_index=True)

# 병합된 val 데이터 저장
val_data.to_csv("/Users/hyunji/Desktop/work/24_2_NLP/Data/A_Data/val_data.csv", index=False, encoding="utf-8-sig")
print("Merged Validation Data saved!")