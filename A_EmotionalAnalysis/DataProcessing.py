import os
import json
import pandas as pd

# 변환할 JSON 파일이 있는 폴더
source_folder = "C:/Users/CBNU/Desktop/이현지/24_2_NLP/Data/A_Data/Training/TL_02.실외"
output_csv = "emotion_dataset_02.csv"

# 데이터 변환
def convert_json_to_csv(folder_path, output_file):
    data = []
    
    # 모든 JSON 파일 순차적으로 처리
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    files.sort(key=lambda x: int(x.split(".")[0]))  # 숫자 순서로 정렬

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
        # JSON 파일 읽기
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        # 데이터 추출
        for conv in json_data["Conversation"]:
            text = conv["Text"]
            label = conv["SpeakerEmotionCategory"]
            data.append({"text": text, "label": label})
    
    # 데이터프레임 생성 및 저장
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding="utf-8-sig", escapechar="\\")
    print(f"Converted data saved to {output_file}")

convert_json_to_csv(source_folder, output_csv)
